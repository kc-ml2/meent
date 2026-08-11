"""Shared machinery for the coefficient cross-validation cases.

Every case in this folder compares the same things in the same way and differs only in its
configuration, so the comparison lives here rather than being copied into each notebook. Four
real bugs were fixed in this logic during its first review; having one copy is the point.

Two normalization modes, chosen by whether the incident wave propagates:

    'flux'  (k_parallel <= n_top)  the incident wave carries z-directed power, so efficiency
                                   is defined and meent's amplitudes can be put on RETICOLO's
                                   energy-flux normalization by multiplying sqrt(f), where
                                   f = Re(kz / (n_top*cos(theta))).

    'raw'   (k_parallel >  n_top)  the incident wave is evanescent and carries no z-flux.
                                   Efficiency has a zero denominator and stops being defined.
                                   The meent sweep writes raw coefficients for diagnostics,
                                   but they are NOT compared directly with RETICOLO's
                                   flux-normalized modal amplitudes. A common incident-field
                                   normalization and a field-basis conversion are required
                                   before that comparison is meaningful.
"""
from dataclasses import dataclass, field
import json

import numpy as np
import torch


# --------------------------------------------------------------------------- configuration

@dataclass
class Case:
    name: str
    # Give EITHER theta_deg (an ordinary real angle) OR k_parallel (in units of k0,
    # which may exceed n_top and then means evanescent input). theta_deg wins if both
    # are set, and k_parallel is derived from it.
    theta_deg: float = None
    k_parallel: float = None
    phi_deg: float = 30.0                # RETICOLO uses angle_delta = -phi_deg
    n_top: float = 1.0
    n_bot: float = 1.0
    # Refractive indices. Scalar, or (nx, ny, nz) for a diagonal tensor. Complex values
    # are written in MEENT's convention, n - i*k, where a POSITIVE k absorbs. RETICOLO uses
    # the opposite sign (its material tables build n + i*k, e.g. retindice.m:1642), so the
    # .m files for a lossy case carry the conjugate index. Feeding either solver the wrong
    # sign turns the medium into a gain medium and R + T comes out above 1 with no error.
    n_grating: object = 2.0
    n_slab: object = 2.0
    n_bg: object = 1.0                   # background the grating sits in
    period: tuple = (1000e-9,)
    thickness: tuple = (500e-9, 500e-9)
    grating_width: float = 500e-9        # 1D
    grating_center: float = 500e-9       # 1D
    grating_dx: float = 500e-9           # 2D pillar
    grating_dy: float = 400e-9           # 2D pillar
    res_xy: tuple = (100, 80)            # 2D raster grid
    fto: tuple = (100,)
    wavelength_m: tuple = (500.5e-9, 700.5e-9, 1e-9)  # general sweep; avoid exact cutoffs
    quick: bool = False
    order_window: int = 6
    type_complex: object = torch.complex128

    tol_selfcheck: float = 1e-12
    tol_mag: dict = field(default_factory=lambda: {
        'discrete': 1e-4, 'continuous': 1e-6, 'vector': 1e-6})
    tol_complex: dict = field(default_factory=lambda: {
        'discrete': 1e-2, 'continuous': 1e-6, 'vector': 1e-6})

    # Relative L2 error per (region, component) in the single-wavelength field comparison.
    # tol_field_edge applies only to the electric component normal to a material edge, which
    # is genuinely discontinuous there and which both solvers reconstruct from a truncated
    # Fourier series; see _field_compare.edge_entries. These two are PROVISIONAL: they are
    # sized from the one case whose field run is recorded in README.md (continuous: 8.6e-12
    # away from the edge, 6.3e-02 for the grating Ex), not from a sweep over all 24.
    tol_field: dict = field(default_factory=lambda: {
        'discrete': 1e-2, 'continuous': 1e-8, 'vector': 1e-8})
    tol_field_edge: dict = field(default_factory=lambda: {
        'discrete': 3e-1, 'continuous': 2e-1, 'vector': 2e-1})

    methods: tuple = ('discrete', 'continuous', 'vector')
    pols: tuple = (0, 1)
    dirs: tuple = ('r', 't')

    @property
    def dim(self):
        # 2 once the grating is periodic in both directions. Note this is the
        # GEOMETRY dimension; an anisotropic 1D ucell is still dim 1 here even
        # though meent routes it through solve_2d.
        return 2 if len(self.period) == 2 else 1

    @property
    def aniso(self):
        return not np.isscalar(self.n_grating)

    @property
    def lossy(self):
        materials = (self.n_bg, self.n_grating, self.n_slab)
        return any(np.any(np.imag(np.asarray(n, dtype=complex)) != 0) for n in materials)

    @property
    def k_par(self):
        if self.theta_deg is not None:
            return self.n_top * np.sin(self.theta_deg * np.pi / 180)
        return self.k_parallel

    @property
    def evanescent(self):
        # Either tangential direction can lie outside the incident-medium light cone.
        return abs(self.k_par) > self.n_top

    @property
    def mode(self):
        return 'raw' if self.evanescent else 'flux'

    @property
    def theta(self):
        # A real angle stays real - important, because meent's complex-theta branch is a
        # different code path and should not be exercised by cases that do not need it.
        if self.theta_deg is not None:
            return self.theta_deg * np.pi / 180
        # complex once k_parallel > n_top: asin of a number above 1 returns pi/2 + i*a,
        # and sin(pi/2 + i*a) = cosh(a) > 1 is exactly the evanescent-input condition.
        return np.arcsin(complex(self.k_parallel / self.n_top, 0))

    @property
    def phi(self):
        return self.phi_deg * np.pi / 180

    @property
    def wavelengths(self):
        lo, hi, step = self.wavelength_m
        sweep_step = step * 20 if self.quick else step
        return np.arange(lo, hi + sweep_step / 2, sweep_step)

    @property
    def orders(self):
        # 1D: [m]. 2D: [(mx, my)] over the window in both directions.
        w = self.order_window
        if self.dim == 1:
            return [(m, 0) for m in range(-w, w + 1)]
        return [(mx, my) for my in range(-w, w + 1) for mx in range(-w, w + 1)]


POL_NAME = {0: 'TE', 1: 'TM'}


# --------------------------------------------------------------------------------- modeling

def diag_material(value, dtype):
    """A scalar or (nx, ny, nz) index as a length-3 diagonal tensor.

    An anisotropic case may still carry a scalar for an isotropic material, and those must
    be broadcast before they can sit in the same array as the tensor ones.
    """
    a = np.asarray(value, dtype=dtype)
    if a.ndim == 0:
        return np.full(3, a, dtype=dtype)
    if a.shape != (3,):
        raise ValueError(f'anisotropic material must be scalar or length 3, got {a.shape}')
    return a


def build_raster(case):
    """(Layers, H, W) isotropic or (Layers, H, W, 3) anisotropic.

    The four geometry families:
      1D  - four cells across the period, the middle two being the grating bar
      2D  - a centred rectangular pillar on a uniform slab, rasterized on res_xy
    """
    dt = complex if case.lossy else float
    if case.dim == 1:
        if case.aniso:
            bg = diag_material(case.n_bg, dt)
            grating = diag_material(case.n_grating, dt)
            slab = diag_material(case.n_slab, dt)
            layer1 = [bg, grating, grating, bg]
            layer2 = [slab] * 4
        else:
            layer1 = [case.n_bg, case.n_grating, case.n_grating, case.n_bg]
            layer2 = [case.n_slab] * 4
        return np.array([[layer1], [layer2]], dtype=dt)

    nx, ny = case.res_xy
    dx, dy = case.period[0] / nx, case.period[1] / ny
    X, Y = np.meshgrid((np.arange(nx) + 0.5) * dx, (np.arange(ny) + 0.5) * dy)
    inside = ((np.abs(X - case.period[0] / 2) <= case.grating_dx / 2) &
              (np.abs(Y - case.period[1] / 2) <= case.grating_dy / 2))
    if case.aniso:
        grating = np.where(inside[..., None], np.array(case.n_grating, dtype=dt),
                           np.array(case.n_bg, dtype=dt))
        slab = np.broadcast_to(np.array(case.n_slab, dtype=dt), grating.shape).copy()
    else:
        grating = np.where(inside, dt(case.n_grating), dt(case.n_bg))
        slab = np.full((ny, nx), dt(case.n_slab))
    return np.stack([grating, slab])


def build_vector(case):
    if case.dim == 1:
        return [
            [case.n_bg, [['rectangle', case.grating_center, case.grating_center,
                          case.grating_width, case.period[0], case.n_grating]]],
            [case.n_slab, []],
        ]
    return [
        [case.n_bg, [['rectangle', case.period[0] / 2, case.period[1] / 2,
                      case.grating_dx, case.grating_dy, case.n_grating]]],
        [case.n_slab, []],
    ]


def make_mee(case, method, pol, wl):
    import meent
    kwargs = dict(backend=2, pol=pol, n_top=case.n_top, n_bot=case.n_bot,
                  fto=list(case.fto), wavelength=wl, period=list(case.period),
                  thickness=list(case.thickness), type_complex=case.type_complex,
                  theta=case.theta, phi=case.phi)
    if method == 'vector':
        return meent.call_mee(ucell=build_vector(case), **kwargs)
    return meent.call_mee(ucell=build_raster(case),
                          fourier_type=0 if method == 'discrete' else 1, **kwargs)


# ---------------------------------------------------------------------------- normalization

def kz_factors(case, mee):
    """sqrt-of-obliquity factors, and kz itself for diagnostics.

    Branch selection mirrors transfer_1d_conical_1 - evanescent orders take the decaying
    branch kz = -1j*sqrt(k_par^2 - n^2). If that changes upstream, this must change with it;
    the self-check is what catches the drift.
    """
    kx, ky = mee.get_kx_ky_vector(wavelength=mee.wavelength)
    k_par2 = kx ** 2 + ky.reshape((-1, 1)) ** 2

    kz_top = (case.n_top ** 2 - k_par2) ** 0.5
    kz_bot = (case.n_bot ** 2 - k_par2) ** 0.5
    ev_t = abs(k_par2) > abs(torch.as_tensor(case.n_top).real ** 2)
    ev_b = abs(k_par2) > abs(torch.as_tensor(case.n_bot).real ** 2)
    kz_top[ev_t] = -1j * (k_par2[ev_t] - case.n_top ** 2) ** 0.5
    kz_bot[ev_b] = -1j * (k_par2[ev_b] - case.n_bot ** 2) ** 0.5

    cos_t = torch.cos(mee.theta)
    f_r = (kz_top / (case.n_top * cos_t)).real
    f_t = (kz_bot / (case.n_top * cos_t)).real
    return f_r, f_t, kz_top, kz_bot


def coefficients(case, mee, res):
    """{'r'|'t': (a_s, a_p)} on the comparison normalization, plus the raw amplitudes."""
    raw = {
        'r': (res.R_s.flatten(), res.R_p.flatten()),
        't': (res.T_s.flatten(), res.T_p.flatten()),
    }
    if case.mode == 'raw':
        return {k: v for k, v in raw.items()}, raw

    f_r, f_t, _, _ = kz_factors(case, mee)
    sr, st = f_r.sqrt().flatten(), f_t.sqrt().flatten()
    norm = {
        'r': (raw['r'][0] * sr, raw['r'][1] * sr),
        't': (raw['t'][0] * st, raw['t'][1] * st),
    }
    return norm, raw


def order_index(case, m):
    """Flat index of diffraction order m = (mx, my) in the (ff_y, ff_x) result array.

    Arrays are flattened row-major, so the y order picks the row and the x order the
    column. In 1D there is a single row and this reduces to fto_x + mx.
    """
    mx, my = m if isinstance(m, tuple) else (m, 0)
    ff_x = 2 * case.fto[0] + 1
    if case.dim == 1:
        return case.fto[0] + mx
    return (case.fto[1] + my) * ff_x + (case.fto[0] + mx)


# ------------------------------------------------------------------------------------- i/o

HEADER = ['wavelength_m', 'order_x', 'order_y',
          'te_re', 'te_im', 'tm_re', 'tm_im', 'efficiency']


def out_name(case, prefix, pol, direction, method=None):
    tail = f'_{method}' if method else ''
    return f'{prefix}_{case.name}_{POL_NAME[pol]}_{direction}{tail}.txt'


def manifest_name(case):
    return f'Meent_{case.name}_manifest.txt'


def _manifest_value(value):
    """JSON-stable representation, including complex scalar/tensor materials."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (tuple, list)):
        return [_manifest_value(v) for v in value]
    if isinstance(value, complex):
        return {'real': value.real, 'imag': value.imag}
    if isinstance(value, np.generic):
        return _manifest_value(value.item())
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def case_manifest(case):
    """Configuration fields that determine the generated coefficient tables."""
    fields = (
        'name', 'theta_deg', 'k_parallel', 'phi_deg', 'n_top', 'n_bot',
        'n_grating', 'n_slab', 'n_bg', 'period', 'thickness', 'grating_width',
        'grating_center', 'grating_dx', 'grating_dy', 'res_xy', 'fto', 'wavelength_m',
        'quick', 'order_window', 'type_complex', 'methods', 'pols', 'dirs',
    )
    return {name: _manifest_value(getattr(case, name)) for name in fields}


def save_manifest(case, workdir):
    p = workdir / manifest_name(case)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(case_manifest(case), f, indent=2, sort_keys=True)
        f.write('\n')


def check_manifest(case, workdir):
    p = workdir / manifest_name(case)
    if not p.exists():
        return f'meent sweep manifest missing: {p.name}'
    try:
        with open(p, encoding='utf-8') as f:
            stored = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return f'meent sweep manifest unreadable: {p.name}: {exc}'
    expected = case_manifest(case)
    if stored != expected:
        changed = sorted(set(stored) | set(expected))
        changed = [k for k in changed if stored.get(k) != expected.get(k)]
        return f'meent sweep manifest does not match this case: {changed}'
    return None


def save_rows(workdir, fname, rows):
    import csv
    with open(workdir / fname, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)


def load_table(workdir, fname):
    """{(wavelength_m, order_x, order_y): (a_te, a_tm, efficiency)}.

    Returns None if the file is absent.

    None and {} are kept distinct on purpose: "never run" must not read as "ran, found
    nothing". RETICOLO files may carry extra columns (raw E/H in the evanescent case);
    only the first eight are read here.
    """
    p = workdir / fname
    if not p.exists():
        return None
    a = np.genfromtxt(p, delimiter=',', skip_header=1)
    if a.size == 0:
        return {}
    if a.ndim == 1:
        a = a.reshape(1, -1)
    out = {}
    for row in a:
        # order_y belongs in the key. Dropping it silently collides (mx, my) with
        # (mx, my') in 2D and the later row wins, losing data without a word.
        key = (round(float(row[0]), 15), int(row[1]), int(row[2]))
        if key in out:
            raise ValueError(f'duplicate coefficient key {key} in {p}')
        out[key] = (
            complex(row[3], row[4]), complex(row[5], row[6]), float(row[7]))
    return out


def row_is_finite(row):
    """Whether both complex amplitudes and the stored efficiency are finite."""
    te, tm, efficiency = row
    return bool(np.all(np.isfinite([
        te.real, te.imag, tm.real, tm.imag, efficiency,
    ])))


def expected_keys(case):
    """Exact (wavelength, order_x, order_y) coverage a meent sweep must write."""
    return {
        (round(float(wl), 15), int(mx), int(my))
        for wl in case.wavelengths
        for mx, my in case.orders
    }


def check_meent_tables(case, workdir):
    """Return coverage problems in saved meent tables, independent of RETICOLO.

    This prevents a reduced smoke sweep from masquerading as a production result when no
    RETICOLO reference exists yet. In particular, mutating ``fto``, ``wavelength_m`` or
    ``order_window`` for a smoke run and writing into a case directory must not pass the
    configured full-sweep verdict.
    """
    expected = expected_keys(case)
    problems = []
    manifest_problem = check_manifest(case, workdir)
    if manifest_problem:
        problems.append(manifest_problem)
    for pol in case.pols:
        for d in case.dirs:
            for method in case.methods:
                fn = out_name(case, 'Meent', pol, d, method)
                table = load_table(workdir, fn)
                if table is None:
                    problems.append(f'meent output missing: {fn}')
                    continue
                got = set(table)
                if got != expected:
                    missing = expected - got
                    extra = got - expected
                    problems.append(
                        f'meent coverage {POL_NAME[pol]} {d} {method}: '
                        f'{len(got)} keys, expected {len(expected)} '
                        f'({len(missing)} missing, {len(extra)} extra)')
    return problems


# ------------------------------------------------------------------------------- the sweep

def run_sweep(case, workdir, verbose=True):
    """Solve meent over the sweep, write one file per (pol, direction, method)."""
    import time
    t0 = time.time()
    failures_sweep, selfcheck = [], []

    # A stale manifest must never bless a partial or differently configured rerun.
    manifest = workdir / manifest_name(case)
    if manifest.exists():
        manifest.unlink()

    for method in case.methods:
        for pol in case.pols:
            rows = {d: [] for d in case.dirs}
            for wl in case.wavelengths:
                try:
                    mee = make_mee(case, method, pol, wl)
                    res = mee.conv_solve().res
                    norm, raw = coefficients(case, mee, res)
                    de = {'r': res.de_ri.flatten(), 't': res.de_ti.flatten()}

                    # The self-check is always done on the FLUX-normalized amplitudes, even
                    # when the comparison runs on raw ones. sum|r_raw*sqrt(f)|^2
                    # == sum|r_raw|^2*f == de is an identity; sum|r_raw|^2 on its own is not,
                    # so checking the raw
                    # amplitudes against de would just report a meaningless mismatch and
                    # lose the drift detection this check exists for.
                    f_r, f_t, _, _ = kz_factors(case, mee)
                    sq = {'r': f_r.sqrt().flatten(), 't': f_t.sqrt().flatten()}

                    for d in case.dirs:
                        a_s, a_p = norm[d]
                        c_s, c_p = raw[d][0] * sq[d], raw[d][1] * sq[d]
                        selfcheck.append((
                            method, pol, wl, d,
                            float((c_s.abs() ** 2 + c_p.abs() ** 2).sum()),
                            float(de[d].sum())))
                        for m in case.orders:
                            i = order_index(case, m)
                            rows[d].append([
                                wl, int(m[0]), int(m[1]),
                                float(a_s[i].real), float(a_s[i].imag),
                                float(a_p[i].real), float(a_p[i].imag),
                                float(de[d][i])])
                except Exception as exc:
                    failures_sweep.append((method, POL_NAME[pol], wl, repr(exc)))
                    for d in case.dirs:
                        for m in case.orders:
                            rows[d].append([wl, int(m[0]), int(m[1])] + [np.nan] * 5)

            for d in case.dirs:
                fname = out_name(case, 'Meent', pol, d, method)
                save_rows(workdir, fname, rows[d])
                if verbose:
                    arr = np.array([r[3] for r in rows[d]], dtype=float)
                    print(f'  {method:11s} {POL_NAME[pol]} {d}  '
                          f'{np.isfinite(arr).sum():5d}/{len(arr):5d} finite  ->  {fname}')

    save_manifest(case, workdir)

    if verbose:
        print(f'\nmeent sweep done in {time.time() - t0:.1f} s, '
              f'{len(failures_sweep)} failed solves')
        for f in failures_sweep[:5]:
            print('   ', f)
        if len(failures_sweep) > 5:
            print(f'    ... and {len(failures_sweep) - 5} more')
    return failures_sweep, selfcheck


def check_selfcheck(case, selfcheck, verbose=True):
    """sum|a|^2 against meent's own efficiencies.

    This is an identity, not physics: (a*sqrt(f))^2 == a^2*f. In 'raw' mode the files retain
    unscaled coefficients, but this check still applies sqrt(f) internally before comparing
    with de. What it verifies is that the factor recomputed in this module matches what the
    solver uses internally - not that an evanescent incident wave has a physical efficiency.
    """
    sc = np.array([(s, d) for _, _, _, _, s, d in selfcheck])
    dev = np.abs(sc[:, 0] - sc[:, 1]) if len(sc) else np.array([])
    if verbose and len(dev):
        print(f'checked  : {len(dev)} (method, pol, wavelength, direction) combinations')
        print(f'max dev  : {dev.max():.3e}')
        print(f'median   : {np.median(dev):.3e}')
        bad = [selfcheck[i] for i in np.where(dev > case.tol_selfcheck)[0]]
        if bad:
            print(f'\n{len(bad)} over {case.tol_selfcheck:g} - the rescaling here is wrong, '
                  f'or kz branch selection drifted from transfer_1d_conical_1:')
            for method, pol, wl, d, s, e in bad[:10]:
                print(f'    {method:11s} {POL_NAME[pol]} {d} {wl:.6g} m  '
                      f'sum|a|^2={s:.12f}  de={e:.12f}')
        else:
            print(f'\nOK - reproduces meent efficiencies to better than '
                  f'{case.tol_selfcheck:g}')
    return dev


# -------------------------------------------------------------------------------- magnitude

def compare_magnitude(case, workdir, verbose=True):
    """|a| per order against RETICOLO. Phase-free, so unambiguous once normalization matches."""
    per_order, unmatched, mag_worst = {}, {}, []

    if case.mode == 'raw':
        if verbose:
            print('SKIP - direct magnitude comparison is undefined for evanescent input:')
            print('       meent raw coefficients and RETICOLO flux-normalized amplitudes do')
            print('       not share a demonstrated incident-field/modal normalization.')
        return per_order, unmatched, mag_worst

    found_any = False

    for pol in case.pols:
        for d in case.dirs:
            ret = load_table(workdir, out_name(case, 'RETICOLO', pol, d))
            if not ret:
                # `not ret` covers both missing and present-but-empty. A header-only file
                # would otherwise give an expected coverage of 0, which every check passes.
                if verbose:
                    what = 'missing' if ret is None else 'EMPTY'
                    print(f'{POL_NAME[pol]:4s} {d:4s}  RETICOLO file {what} - run the .m file')
                continue
            found_any = True
            for method in case.methods:
                me = load_table(workdir, out_name(case, 'Meent', pol, d, method))
                if not me:
                    continue
                me_lams = {k[0] for k in me}
                for key, ret_row in ret.items():
                    lam, mx, my = key
                    m = (mx, my)
                    if key not in me:
                        # A wavelength meent never solved is a different problem from an
                        # order outside ORDER_WINDOW, and the fixes differ.
                        reason = 'order' if lam in me_lams else 'wavelength'
                        unmatched.setdefault((pol, d, method, reason), set()).add(
                            m if reason == 'order' else lam)
                        continue
                    me_row = me[key]
                    if not (row_is_finite(ret_row) and row_is_finite(me_row)):
                        unmatched.setdefault((pol, d, method, 'nonfinite'), set()).add(key)
                        continue
                    rte, rtm, _ = ret_row
                    mte, mtm, _ = me_row
                    dev = max(abs(abs(mte) - abs(rte)), abs(abs(mtm) - abs(rtm)))
                    per_order.setdefault((pol, d, method, m), []).append(dev)
                    mag_worst.append((dev, pol, d, method, lam, m))

    if verbose:
        if found_any:
            print(f"{'pol':4s} {'dir':4s} {'method':11s} "
                  f"{'max d|a|':>11s} {'median':>11s} {'n':>6s}")
            print('-' * 52)
            for pol in case.pols:
                for d in case.dirs:
                    for method in case.methods:
                        vals = [v for (p_, d_, m_, o_), lst in per_order.items()
                                if (p_, d_, m_) == (pol, d, method) for v in lst]
                        if not vals:
                            continue
                        tol = case.tol_mag[method]
                        over = sum(1 for v in vals if v > tol)
                        mark = f'  <-- {over} over {tol:g}' if over else ''
                        print(f'{POL_NAME[pol]:4s} {d:4s} {method:11s} '
                              f'{max(vals):11.3e} {np.median(vals):11.3e} '
                              f'{len(vals):6d}{mark}')
        else:
            print('No RETICOLO reference found. The meent-only checks still apply.')
    return per_order, unmatched, mag_worst


def report_per_order(case, per_order, unmatched):
    """The summary above is a max over all orders, which is how one bad order hides."""
    if per_order:
        orders_seen = sorted({k[3] for k in per_order})
        lab = (lambda m: f'{m[0]:+d}' if case.dim == 1 else f'{m[0]:+d},{m[1]:+d}')
        for pol in case.pols:
            for d in case.dirs:
                rows = [(mth, [per_order.get((pol, d, mth, m)) for m in orders_seen])
                        for mth in case.methods]
                rows = [(mth, v) for mth, v in rows if any(x for x in v)]
                if not rows:
                    continue
                print(f'--- {POL_NAME[pol]} incidence, {d} ---')
                print('   order ' + ''.join(f'{lab(m):>12s}' for m in orders_seen))
                for mth, vals in rows:
                    cells = [f'{max(v):12.2e}' if v else f'{"-":>12s}' for v in vals]
                    print(f'{mth:>8s} ' + ''.join(cells))
                print()
    else:
        print('nothing to break down - no RETICOLO reference loaded')

    by_reason = {}
    for (pol, d, method, reason), vals in unmatched.items():
        by_reason.setdefault(reason, set()).update(vals)
    if by_reason.get('order'):
        print(f"RETICOLO orders outside ORDER_WINDOW=+-{case.order_window}, not compared: "
              f"{sorted(by_reason['order'])}  -> raise order_window")
    if by_reason.get('wavelength'):
        wl_ = sorted(by_reason['wavelength'])
        print(f'{len(wl_)} RETICOLO wavelengths meent never solved '
              f'({wl_[0]:.6g}-{wl_[-1]:.6g} m)  -> quick run, or the sweep failed there')
    if by_reason.get('nonfinite'):
        print(f'{len(by_reason["nonfinite"])} unique matched keys contained NaN or Inf '
              f'in one or more tables and were not compared')
    if not unmatched:
        print('every RETICOLO (wavelength, order) had a meent counterpart')


# ------------------------------------------------------------------------------------ phase

AMP_FLOOR = 1e-8


def jones_vector(table, lam, floor=AMP_FLOOR, orders=None):
    """TE/TM-interleaved vector and its orders.

    With ``orders=None`` this selects usable orders from the reference table. Passing an
    explicit order list aligns the other solver to that reference set. The latter must not
    independently apply the amplitude floor: doing so can silently skip a real disagreement
    near the floor, and raw meent files may contain nonzero evanescent orders that res2 does
    not report.
    """
    requested_orders = orders
    vec = []
    candidate_orders = (sorted(k[1:] for k in table if k[0] == lam)
                        if requested_orders is None else list(requested_orders))
    selected = []
    for m in candidate_orders:
        row = table.get((lam,) + tuple(m))
        if row is None:
            return np.array([]), []
        if not row_is_finite(row):
            return np.array([]), []
        te, tm, _ = row
        if requested_orders is None and max(abs(te), abs(tm)) < floor:
            continue
        vec += [te, tm]
        selected.append(m)
    return np.array(vec), selected


def compare_phase(case, workdir, verbose=True):
    """Direct complex-amplitude comparison in the shared RETICOLO convention.

    Nothing is fitted here - no phase, sign, amplitude, or conjugation. What goes into the
    compared vector is still deliberate:

      * TE and TM together - comparing each component against its own zeroth order would
        hide a constant offset between them, which is the cross-polarization term that
        matters under conical incidence.
      * r and t together - they come from one incident wave, so a constant phase error on
        every transmitted order must remain visible.

    The conjugated vector and the common phase ``alpha`` are computed as diagnostics for a
    time-convention regression only. Neither is ever substituted into the reported error.
    """
    phase_worst, conj_votes, phase_n = [], [], {}

    if case.mode == 'raw':
        if verbose:
            print('SKIP - direct complex-amplitude comparison is undefined for evanescent input.')
        return phase_worst, conj_votes, phase_n

    for pol in case.pols:
        for method in case.methods:
            tabs, ok = {}, True
            for d in case.dirs:
                ret = load_table(workdir, out_name(case, 'RETICOLO', pol, d))
                me = load_table(workdir, out_name(case, 'Meent', pol, d, method))
                if not ret or not me:
                    ok = False
                    break
                tabs[d] = (ret, me)
            if not ok:
                continue

            errs = []
            for lam in sorted({k[0] for k in tabs[case.dirs[0]][0]}):
                rv_all, mv_all, good = [], [], True
                for d in case.dirs:
                    ret, me = tabs[d]
                    rv, ro = jones_vector(ret, lam)
                    mv, mo = jones_vector(me, lam, orders=ro)
                    if ro != mo or len(rv) == 0 or not np.all(np.isfinite(mv)):
                        good = False
                        break
                    rv_all.append(rv)
                    mv_all.append(mv)
                if not good:
                    continue
                rv, mv = np.concatenate(rv_all), np.concatenate(mv_all)

                # No phase, sign, conjugation, or amplitude fit is allowed here. Meent has
                # already converted its public coefficients to RETICOLO's convention; the
                # validation is therefore the direct complex difference.
                err = np.abs(mv - rv).max()
                err_c = np.abs(np.conj(mv) - rv).max()
                conj_votes.append(err <= err_c)
                errs.append(err)
                # Keep the measured common phase only as a diagnostic. It is not removed.
                alpha = np.angle(np.vdot(rv, mv))
                phase_worst.append((err, pol, method, lam, alpha))

            phase_n[(pol, method)] = len(errs)
            if verbose and errs:
                tol = case.tol_complex[method]
                over = sum(1 for v in errs if v > tol)
                mark = f'  <-- {over} over {tol:g}' if over else ''
                print(f'{POL_NAME[pol]:4s} {method:11s} '
                      f'{max(errs):11.3e} {np.median(errs):11.3e} {len(errs):6d}{mark}')

    if verbose and conj_votes:
        frac = np.mean(conj_votes)
        print()
        if frac > 0.99:
            print(f'Time convention: matched as-is ({frac:.1%} of wavelengths).')
        elif frac < 0.01:
            print(f'Time convention: CONJUGATE ({1-frac:.1%}). On its own that is a '
                  f'convention difference - and it holds for r and t together here, '
                  f'because they are compared together.')
        else:
            print(f'Time convention: SPLIT ({frac:.1%} straight) - genuine disagreement, '
                  f'not a convention.')
    return phase_worst, conj_votes, phase_n


# ------------------------------------------------------------------------------------ plots

def plot_coefficients(case, workdir, pol=0, component=0, order_limit=2,
                      show_reference=True):
    """Re and Im of the complex coefficient per order against wavelength.

    Not |a|. A magnitude plot hides exactly what this directory exists to catch: two
    solvers can agree on every magnitude and still differ by a sign, a conjugation, or a
    per-order reference-plane phase. The verdict is a direct complex difference, so the
    plot shows the same numbers the verdict is computed from - real and imaginary parts
    exactly as stored, with no absolute value taken anywhere.

    ``show_reference=False`` for the evanescent case, where meent raw coefficients and
    RETICOLO flux-normalized amplitudes do not share a normalization, so an overlay would
    draw two different quantities on one axis.
    """
    import matplotlib.pyplot as plt

    show = [m for m in case.orders
            if abs(m[0]) <= order_limit and abs(m[1]) <= (0 if case.dim == 1 else 1)]
    if not show:
        print('no orders within order_limit to plot')
        return None

    ret_tables, meent_tables = {}, {}
    for d in case.dirs:
        ret_tables[d] = (load_table(workdir, out_name(case, 'RETICOLO', pol, d))
                         if show_reference else None)
        for method in case.methods:
            meent_tables[(d, method)] = load_table(
                workdir, out_name(case, 'Meent', pol, d, method))

    styles = {'discrete': '--', 'continuous': ':', 'vector': '-.'}
    parts = (('Re', np.real), ('Im', np.imag))
    omark = (lambda m: f'{m[0]:+d}' if case.dim == 1 else f'{m[0]:+d},{m[1]:+d}')

    ncol = len(case.dirs) * len(parts)
    fig, axes = plt.subplots(len(show), ncol, squeeze=False, sharex=True,
                             figsize=(3.4 * ncol, 2.6 * len(show)))
    for r_i, m in enumerate(show):
        for d_i, d in enumerate(case.dirs):
            for p_i, (pname, take) in enumerate(parts):
                ax = axes[r_i, d_i * len(parts) + p_i]
                for method in case.methods:
                    table = meent_tables[(d, method)]
                    if not table:
                        continue
                    lam = sorted({k[0] for k in table if k[1:] == m})
                    if lam:
                        ax.plot(lam, [take(table[(l,) + m][component]) for l in lam],
                                styles[method], label=f'meent {method}')
                ret = ret_tables[d]
                if ret:
                    lam = sorted({k[0] for k in ret if k[1:] == m})
                    if lam:
                        ax.plot(lam, [take(ret[(l,) + m][component]) for l in lam],
                                'k-', lw=2, label='RETICOLO')
                # Re and Im are signed, so zero is a meaningful line here in a way it
                # never was on a |a| axis.
                ax.axhline(0, color='0.7', lw=.6, zorder=0)
                ax.set_title(f'{d}, order {omark(m)}, {POL_NAME[component]} component, '
                             f'{pname}', fontsize=9)
                ax.set_ylabel(f'{pname}(a)')
                ax.grid(alpha=.3)
    for ax in axes[-1]:
        ax.set_xlabel('wavelength (m)')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f'{case.name} - {POL_NAME[pol]} incidence, complex coefficient', y=1.0)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------- verdict

def verdict(case, workdir, failures_sweep, dev, per_order, unmatched,
            phase_worst, phase_n):
    """Collect every check and raise. A run that ends without an exception is the pass."""
    failures = []

    missing, empty = [], []
    for pol in case.pols:
        for d in case.dirs:
            fn = out_name(case, 'RETICOLO', pol, d)
            t = load_table(workdir, fn)
            if t is None:
                missing.append(fn)
            elif not t:
                empty.append(fn)
    if missing:
        failures.append(f'RETICOLO reference missing: {missing}')
    if empty:
        # A header-only file makes every expected count zero, so the coverage checks
        # below would all pass vacuously.
        failures.append(f'RETICOLO reference present but empty: {empty}')

    if failures_sweep:
        failures.append(f'{len(failures_sweep)} meent solves failed, '
                        f'first: {failures_sweep[0]}')

    failures.extend(check_meent_tables(case, workdir))

    if case.mode == 'raw':
        failures.append(
            'evanescent-input cross-solver comparison is not implemented: raw meent '
            'coefficients cannot be compared directly with RETICOLO flux-normalized '
            'amplitudes; normalize physical E/H to a common incident field first')

    skipped_orders = {m for (p_, d_, mt_, r_), v in unmatched.items()
                      if r_ == 'order' for m in v}
    if skipped_orders:
        failures.append(f'orders outside order_window, never compared: '
                        f'{sorted(skipped_orders)}')
    skipped_wl = {w for (p_, d_, mt_, r_), v in unmatched.items()
                  if r_ == 'wavelength' for w in v}
    if skipped_wl:
        why = ' (quick=True solves a subset)' if case.quick else ''
        failures.append(f'{len(skipped_wl)} RETICOLO wavelengths never solved{why}')
    nonfinite = {(p_, d_, mt_, key)
                 for (p_, d_, mt_, r_), values in unmatched.items()
                 if r_ == 'nonfinite' for key in values}
    if nonfinite:
        failures.append(f'{len(nonfinite)} matched coefficient rows contained NaN or Inf')

    if dev is None or not len(dev):
        failures.append('normalization self-check did not run')
    elif dev.max() > case.tol_selfcheck:
        failures.append(f'self-check {dev.max():.3e} > {case.tol_selfcheck:g}')

    # Raw evanescent-input files are diagnostic data, not cross-solver evidence. Stop here
    # instead of adding a long list of expected coverage failures for comparisons that were
    # deliberately disabled above.
    if case.mode == 'raw':
        print(f'{len(failures)} failure(s)')
        for f in failures:
            print('  -', f)
        assert not failures, f'{len(failures)} validation failure(s) - see above'

    # Coverage. The loops above skip past empty files, NaN and mismatched order sets;
    # without this those look exactly like "compared and agreed".
    for pol in case.pols:
        for d in case.dirs:
            ret = load_table(workdir, out_name(case, 'RETICOLO', pol, d))
            if not ret:
                continue
            for method in case.methods:
                got = sum(len(v) for (p_, d_, m_, o_), v in per_order.items()
                          if (p_, d_, m_) == (pol, d, method))
                if got != len(ret):
                    failures.append(f'|a| coverage {POL_NAME[pol]} {d} {method}: '
                                    f'compared {got} of {len(ret)} pairs')
        ret0 = load_table(workdir, out_name(case, 'RETICOLO', pol, case.dirs[0]))
        if not ret0:
            continue
        expect_wl = len({k[0] for k in ret0})
        for method in case.methods:
            got_wl = phase_n.get((pol, method))
            if got_wl is None:
                failures.append(f'phase comparison never ran for {POL_NAME[pol]} {method}')
            elif got_wl != expect_wl:
                failures.append(f'phase coverage {POL_NAME[pol]} {method}: '
                                f'compared {got_wl} of {expect_wl} wavelengths')

    for (pol, d, method, m), vals in per_order.items():
        tol = case.tol_mag[method]
        if max(vals) > tol:
            failures.append(f'|a| {POL_NAME[pol]} {d} {method} order {m}: '
                            f'{max(vals):.3e} > {tol:g}')

    worst_by = {}
    for v, pol, method, lam, _ in phase_worst:
        worst_by[(pol, method)] = max(worst_by.get((pol, method), 0.0), v)
    for (pol, method), v in worst_by.items():
        tol = case.tol_complex[method]
        if v > tol:
            failures.append(f'complex {POL_NAME[pol]} {method}: {v:.3e} > {tol:g}')
    if not phase_worst:
        failures.append('phase comparison did not run')

    print(f'{len(failures)} failure(s)')
    for f in failures:
        print('  -', f)
    assert not failures, f'{len(failures)} validation failure(s) - see above'
    print()
    print('PASS - meent agrees with RETICOLO on every checked coefficient.')
