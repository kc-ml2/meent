#!/usr/bin/env python3
"""Re-verify this reference case against a checkout where anisotropy has merged.

    python tests/reference_cases/1D_anisotropic_grating_conical_incidence/verify_after_merge.py

Run this FIRST, before `pytest`. The anisotropy PR was written by someone else
and may not match the branch this case was built against, so this script checks
each API assumption `case.py` depends on *individually* and tells you which one
moved -- rather than handing you a wall of pytest failures that all trace back
to one changed signature.

It changes nothing. It prints a report and, at the end, the exact values to
paste into case.py.

See RERUN.md in this directory for what to do with the output.
"""

import sys
import time
import traceback
from pathlib import Path

import numpy as np

CASE_DIR = Path(__file__).resolve().parent
TESTS_DIR = CASE_DIR.parent.parent
REPO_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(TESTS_DIR))

from reference_cases import _loader  # noqa: E402

case = _loader.load_case(CASE_DIR)

PASS, FAIL, SKIP = 'PASS', 'FAIL', 'SKIP'
results = []


def record(name, status, detail=''):
    results.append((name, status, detail))
    mark = {PASS: ' ok ', FAIL: 'FAIL', SKIP: 'skip'}[status]
    print(f'  [{mark}] {name}' + (f'\n         {detail}' if detail else ''))
    return status == PASS


def check(name, fn):
    """Run one contract check in isolation; any exception is a FAIL, not a crash."""
    try:
        ok, detail = fn()
        return record(name, PASS if ok else FAIL, detail)
    except Exception as exc:  # noqa: BLE001
        return record(name, FAIL, f'{type(exc).__name__}: {exc}')


def header(text):
    print(f'\n{text}\n' + '-' * len(text))


# --------------------------------------------------------------------------- #
header('0. environment')
# --------------------------------------------------------------------------- #

try:
    import meent
    print(f'  meent      {getattr(meent, "__file__", "?")}')
except Exception as exc:  # noqa: BLE001
    print(f'  meent      NOT IMPORTABLE: {exc}')
    print('\nCannot continue. Install meent (pip install -e .) and re-run.')
    sys.exit(2)

try:
    import torch
    print(f'  torch      {torch.__version__}')
except Exception:  # noqa: BLE001
    print('  torch      NOT INSTALLED -- this case is torch-only, nothing below will run')
    sys.exit(2)

print(f'  numpy      {np.__version__}')

try:
    import subprocess
    for label, args in (('branch', ['rev-parse', '--abbrev-ref', 'HEAD']),
                        ('commit', ['rev-parse', '--short', 'HEAD'])):
        out = subprocess.run(['git', '-C', str(REPO_ROOT), *args],
                             capture_output=True, text=True).stdout.strip()
        print(f'  {label:10s} {out}')
except Exception:  # noqa: BLE001
    pass

# --------------------------------------------------------------------------- #
header('1. API contract -- what case.py assumes about anisotropy')
# --------------------------------------------------------------------------- #
# Each check states the assumption. A FAIL here means the merged PR differs from
# the branch this case was built against; RERUN.md says what to do about each.

WL = float(case.SMOKE_WAVELENGTHS_NM[0])


def _build(method, pol=0, wavelength_nm=WL):
    return meent.call_mee(**case.build(method, pol, wavelength_nm))


def c1():
    """A 4-D ucell (layers, H, W, 3) is accepted as a diagonal (nx, ny, nz) tensor."""
    mee = _build('discrete')
    shape = tuple(mee.ucell.shape)
    return shape[-1] == 3, f'ucell shape {shape}, expected trailing 3'


def c2():
    """An anisotropic conical case solves without raising."""
    result = _build('discrete').conv_solve()
    r = float(result.res.de_ri.sum())
    t = float(result.res.de_ti.sum())
    return np.isfinite(r) and np.isfinite(t), f'R = {r:.9f}, T = {t:.9f}'


def c3():
    """Vector instructions accept a 3-component index."""
    result = _build('vector').conv_solve()
    return np.isfinite(float(result.res.de_ri.sum())), 'vector solve returned a finite R'


def c4():
    """Anisotropic + conical routes to the 2D solver (grating_type_assigned == 2)."""
    mee = _build('discrete')
    mee.conv_solve()
    gt = mee.grating_type_assigned
    return gt == 2, f'grating_type_assigned = {gt}, expected 2'


def c5():
    """type_complex=0 still means complex128."""
    mee = _build('discrete')
    return mee.type_complex in (torch.complex128,), f'type_complex = {mee.type_complex}'


def c6():
    """Isotropic ucells still work -- the PR must not break the ordinary path."""
    ucell = np.array([[[1.0, 2.0, 2.0, 1.0]]])
    mee = meent.call_mee(backend=2, pol=0, n_top=1, n_bot=1, theta=0.1, phi=None,
                         fto=[5], wavelength=500e-9, period=[1000e-9],
                         thickness=[500e-9], ucell=ucell, type_complex=0)
    result = mee.conv_solve()
    r = float(result.res.de_ri.sum())
    t = float(result.res.de_ti.sum())
    return abs(r + t - 1) < 1e-9, f'isotropic 1D: R + T - 1 = {r + t - 1:.3e}'


def c7():
    """A single conical solve also exposes res_te_inc / res_tm_inc."""
    result = _build('discrete').conv_solve()
    have = result.res_te_inc is not None and result.res_tm_inc is not None
    return have, 'both TE and TM incidence channels present' if have else 'missing'


def c8():
    """pol=0 (.res) equals the res_te_inc channel of the same solve.

    Not required by case.py -- but if it holds, the sweep can be halved by
    solving once per wavelength instead of once per polarization.
    """
    result = _build('discrete', pol=0).conv_solve()
    if result.res_te_inc is None:
        return False, 'res_te_inc is None'
    a = float(result.res.de_ri.sum())
    b = float(result.res_te_inc.de_ri.sum())
    return abs(a - b) < 1e-12, f'.res R = {a:.12f} vs res_te_inc R = {b:.12f}'


for name, fn in [
    ('4-D anisotropic ucell accepted', c1),
    ('anisotropic conical solve runs', c2),
    ('vector instruction takes a 3-component index', c3),
    ('routes to the 2D solver (grating type 2)', c4),
    ('type_complex=0 is complex128', c5),
    ('isotropic path still works', c6),
    ('TE/TM incidence channels present', c7),
    ('pol=0 .res == res_te_inc (optimization, optional)', c8),
]:
    check(name, fn)

contract_ok = all(s == PASS for n, s, _ in results if 'optional' not in n)
if not contract_ok:
    print('\n  >>> The API moved. Read RERUN.md section "If a contract check fails"')
    print('  >>> before trusting any number below.')

# --------------------------------------------------------------------------- #
header('2. numeric agreement on the smoke wavelengths')
# --------------------------------------------------------------------------- #

print(f'  {len(case.SMOKE_WAVELENGTHS_NM)} wavelengths x {len(case.METHODS)} methods '
      f'x {len(case.POLS)} polarizations = '
      f'{len(case.SMOKE_WAVELENGTHS_NM) * len(case.METHODS) * len(case.POLS)} solves\n')

worst_vs_recorded = {m: 0.0 for m in case.METHODS}
worst_vs_reference = {m: 0.0 for m in case.METHODS}
worst_energy = 0.0
n_solved = 0
t0 = time.time()

print(f'  {"pol":4s} {"method":11s} {"lambda":>8s} {"|R+T-1|":>11s} '
      f'{"vs recorded":>13s} {"vs RETICOLO":>13s}')
print('  ' + '-' * 64)

for pol in case.POLS:
    reference = _loader.reference_table(case, pol)
    for method in case.METHODS:
        recorded = _loader.recorded_table(case, pol, method)
        for wl in case.SMOKE_WAVELENGTHS_NM:
            try:
                r, t = case.observables(_build(method, pol, wl).conv_solve())
            except Exception:  # noqa: BLE001
                print(f'  {case.POLS[pol]:4s} {method:11s} {wl:8.0f}  SOLVE FAILED')
                traceback.print_exc(limit=1)
                continue
            n_solved += 1

            row_rec = recorded[np.isclose(recorded[:, 0], wl, atol=1e-6)][0]
            row_ref = reference[np.isclose(reference[:, 0], wl, atol=1e-6)][0]
            d_rec = max(abs(r - row_rec[1]), abs(t - row_rec[2]))
            d_ref = max(abs(r - row_ref[1]), abs(t - row_ref[2]))
            energy = abs(r + t - 1)

            worst_vs_recorded[method] = max(worst_vs_recorded[method], d_rec)
            worst_vs_reference[method] = max(worst_vs_reference[method], d_ref)
            worst_energy = max(worst_energy, energy)

            flag = ''
            if d_ref > case.TOL_REFERENCE[method]:
                flag = f'  <-- over TOL_REFERENCE ({case.TOL_REFERENCE[method]:g})'
            print(f'  {case.POLS[pol]:4s} {method:11s} {wl:8.0f} {energy:11.3e} '
                  f'{d_rec:13.3e} {d_ref:13.3e}{flag}')

print(f'\n  {n_solved} solves in {time.time() - t0:.1f} s')

# --------------------------------------------------------------------------- #
header('3. verdict')
# --------------------------------------------------------------------------- #


def decade_above(x):
    """Next power of ten strictly above x, with a floor at 1e-15."""
    if x <= 0:
        return 1e-12
    return float(10.0 ** np.ceil(np.log10(x) + 0.5))


energy_ok = worst_energy <= case.TOL_ENERGY
print(f'  energy conservation   worst |R+T-1| = {worst_energy:.3e}  '
      f'(tolerance {case.TOL_ENERGY:g})  {"ok" if energy_ok else "OVER TOLERANCE"}')

ref_ok = True
for method in case.METHODS:
    over = worst_vs_reference[method] > case.TOL_REFERENCE[method]
    ref_ok &= not over
    print(f'  vs RETICOLO  {method:11s} {worst_vs_reference[method]:.3e}  '
          f'(tolerance {case.TOL_REFERENCE[method]:g})  {"OVER" if over else "ok"}')

print()
print('  vs the recorded run -- this is the number TOL_RECORDED was waiting for:')
for method in case.METHODS:
    print(f'    {method:11s} {worst_vs_recorded[method]:.3e}')
suggested = decade_above(max(worst_vs_recorded.values()))
print(f'\n  Suggested TOL_RECORDED = {suggested:g}   '
      f'(worst {max(worst_vs_recorded.values()):.3e}, rounded up ~half a decade)')

print()
if contract_ok and energy_ok and ref_ok:
    print('  VERDICT: the merged checkout reproduces this case.')
    print('  Next: set TOL_RECORDED in case.py, then run')
    print('        pytest tests/test_reference.py -v')
    print('        pytest tests/test_reference.py -m slow      # full 201-point sweep')
else:
    print('  VERDICT: something moved. Do NOT widen a tolerance to make this pass.')
    print('  Read RERUN.md -- it lists what each failure means and what to do.')

print()
print('  Paste into case.py when satisfied:')
print(f'    TOL_RECORDED = {suggested:g}')
print('    MEASURED_REFERENCE = {')
for method in case.METHODS:
    print(f'        {method!r}: {worst_vs_reference[method]:.2e},'
          f'   # smoke subset only -- run -m slow for the full sweep figure')
print('    }')

sys.exit(0 if (contract_ok and energy_ok and ref_ok) else 1)
