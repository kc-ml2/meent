"""Single-wavelength spatial-field comparison between Meent and RETICOLO.

The physical structure is padded with homogeneous top and bottom layers whose indices equal
the semi-infinite media.  This makes both solvers expose the superstrate/substrate fields as
ordinary finite-layer fields without adding an optical interface.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

import _compare as C


COMPONENTS = ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')
POL_NAMES = ('TE', 'TM')
REGIONS = ('superstrate', 'grating', 'slab', 'substrate')
def wavelength_tag(wavelength_m):
    text = format(float(wavelength_m), '.12g')
    return text.replace('.', 'p')


def reticolo_field_name(case, wavelength_m):
    return f'RETICOLO_{case.name}_field_{wavelength_tag(wavelength_m)}m.mat'


def write_field_request(workdir, wavelength_m):
    """Write the single wavelength consumed by the matching MATLAB field script."""
    wavelength_m = float(wavelength_m)
    if not np.isfinite(wavelength_m) or wavelength_m <= 0:
        raise ValueError(f'wavelength must be positive and finite, got {wavelength_m}')
    path = Path(workdir) / 'field_wavelength_m.txt'
    path.write_text(f'{wavelength_m:.17g}\n', encoding='ascii')
    print(f'field request written: {path.name} = {wavelength_m:.6g} m')
    return path


def _as_numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
    return np.asarray(value)


def _uniform_raster_layer(base, value):
    """A homogeneous layer with the same spatial/tensor shape as one raster layer."""
    dtype = base.dtype
    if base.ndim == 4:  # layers, y, x, diagonal tensor
        layer = np.empty(base.shape[1:], dtype=dtype)
        layer[...] = C.diag_material(value, dtype)
        return layer
    return np.full(base.shape[1:], value, dtype=dtype)


def build_field_model(case, method, buffer_top, buffer_bottom):
    """Return ucell and thickness with index-matched finite exterior buffers."""
    if method == 'vector':
        ucell = [[case.n_top, []], *C.build_vector(case), [case.n_bot, []]]
    else:
        base = C.build_raster(case)
        top = _uniform_raster_layer(base, case.n_top)
        bottom = _uniform_raster_layer(base, case.n_bot)
        ucell = np.concatenate(([top], base, [bottom]), axis=0)
    thickness = (float(buffer_top), *map(float, case.thickness), float(buffer_bottom))
    return ucell, thickness


def make_field_mee(case, method, pol, wavelength, buffer_top, buffer_bottom):
    import meent

    if method not in case.methods:
        raise ValueError(f'unknown field method {method!r}; choose one of {case.methods}')
    ucell, thickness = build_field_model(case, method, buffer_top, buffer_bottom)
    kwargs = dict(
        backend=2, pol=pol, n_top=case.n_top, n_bot=case.n_bot,
        fto=list(case.fto), wavelength=wavelength, period=list(case.period),
        thickness=list(thickness), type_complex=case.type_complex,
        theta=case.theta, phi=case.phi, ucell=ucell,
    )
    if method != 'vector':
        kwargs['fourier_type'] = 0 if method == 'discrete' else 1
    return meent.call_mee(**kwargs)


def _six_components(field):
    """Meent's public field as a NumPy (z, y, x, Ex..Hz) array.

    ``RCWATorch.calculate_field`` already expands the scalar 1D solver's three-component
    output to Ex..Hz *and* applies the RETICOLO convention to it. Re-expanding a
    three-component field here would produce an unconverted array - no conjugation, no
    component signs, no scalar-1D TM sign - that ``compare_field_profile`` would then
    subtract from RETICOLO directly. So a component count other than six is a solver
    regression to raise on, never something to repair in the validation layer.
    """
    field = _as_numpy(field)
    if field.ndim == 5:  # selected incident state, z, y, x, component
        if field.shape[0] != 1:
            raise ValueError(f'expected one selected incident field, got shape {field.shape}')
        field = field[0]
    if field.ndim != 4:
        raise ValueError(f'unexpected Meent field shape {field.shape}')
    if field.shape[-1] != 6:
        raise ValueError(
            f'calculate_field returned {field.shape[-1]} components, expected 6 (Ex..Hz) '
            f'already in RETICOLO convention')
    return field


def calculate_meent_fields(case, wavelength_m, method, res_x, res_y, res_z,
                           buffer_top, buffer_bottom):
    fields = []
    wavelength = float(wavelength_m)
    for pol in (0, 1):
        mee = make_field_mee(case, method, pol, wavelength, buffer_top, buffer_bottom)
        mee.conv_solve()
        field = mee.calculate_field(
            res_x=res_x,
            res_y=res_y if case.dim == 2 else 1,
            res_z=res_z,
            set_field_input=(True, False, False),
        )
        # Meent itself returns Ex..Hz in RETICOLO's convention. This helper only converts
        # the tensor object to NumPy and validates the public component shape.
        fields.append(_six_components(field))
    return np.stack(fields)


def _mat_scalar(data, name):
    return float(np.asarray(data[name]).squeeze())


def _mat_text(data, name):
    return str(np.asarray(data[name]).squeeze())


def load_reticolo_fields(case, workdir, wavelength_m):
    path = Path(workdir) / reticolo_field_name(case, wavelength_m)
    if not path.is_file():
        field_script = Path(workdir) / f'RETICOLO_{case.name}.m'
        raise FileNotFoundError(
            f'{path.name} is missing. Set FIELD_WAVELENGTH_M={wavelength_m:.6g} in '
            f'{field_script.name}, run that MATLAB script, then rerun this cell.')

    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    stored_case = _mat_text(data, 'case_name')
    if stored_case != case.name:
        raise ValueError(f'{path.name} stores case {stored_case!r}, expected {case.name!r}')
    stored_wavelength = _mat_scalar(data, 'wavelength_m')
    if not np.isclose(stored_wavelength, wavelength_m, rtol=0, atol=1e-15):
        raise ValueError(
            f'{path.name} stores {stored_wavelength:.6g} m, expected {wavelength_m:.6g} m')

    converted = []
    for name in ('field_TE', 'field_TM'):
        field = np.asarray(data[name])
        if case.dim == 1:
            if field.ndim == 4 and field.shape[2] == 1:
                field = field[:, :, 0, :]
            if field.ndim != 3:
                raise ValueError(f'unexpected RETICOLO 1D field shape {field.shape}')
            field = field[:, None, :, :]
        else:
            if field.ndim != 4:
                raise ValueError(f'unexpected RETICOLO 2D field shape {field.shape}')
            # RETICOLO: z,x,y,c; Meent: z,y,x,c.
            field = field.swapaxes(1, 2)
        # RETICOLO orders z bottom-to-top. Flip only the array layout so both arrays index
        # the same physical sample; do not conjugate, change signs, or otherwise alter its
        # stored field values. All convention conversion is applied to Meent above.
        converted.append(np.flip(field, axis=0))

    profile_heights = np.asarray(data['profile_heights'], dtype=float).reshape(-1)
    return {
        'path': path,
        'field': np.stack(converted),
        'profile_heights': profile_heights,
        'res_x': int(_mat_scalar(data, 'res_x')),
        'res_y': int(_mat_scalar(data, 'res_y')),
        'res_z': int(_mat_scalar(data, 'res_z')),
        'x': np.asarray(data['x'], dtype=float).reshape(-1),
        'y': np.asarray(data['y'], dtype=float).reshape(-1),
        'z_raw': np.asarray(data['z'], dtype=float).reshape(-1),
    }


def _z_coordinates(thickness, res_z):
    blocks, offset = [], 0.0
    for height in thickness:
        blocks.append(offset + np.linspace(0, height, res_z))
        offset += height
    return np.concatenate(blocks)


def _region_slices(res_z, n_layers):
    return [slice(i * res_z, (i + 1) * res_z) for i in range(n_layers)]


def check_grid(case, reference, thickness):
    """RETICOLO's stored sample coordinates against the grid Meent will evaluate on.

    Nothing else compares them: ``compare_field_profile`` only checks that the two arrays
    have the same shape, and shapes agree whenever the point counts do. If either solver
    changed where it places samples inside a layer, the codes would be compared at
    different points and the result would read as an ordinary numerical error rather than
    as a grid mismatch. The checks are free - both grids agree exactly today.
    """
    res_x, res_y, res_z = reference['res_x'], reference['res_y'], reference['res_z']

    # meent/on_torch/emsolver/field_distribution.py: linspace(0, period[0], res_x).
    x = np.linspace(0, case.period[0], res_x)
    if not np.allclose(reference['x'], x, rtol=0, atol=case.period[0] * 1e-12):
        raise ValueError('RETICOLO x samples do not match the Meent x grid')

    if case.dim == 2:
        # Meent flips its y axis, so its grid runs from period_y down to zero.
        y = np.linspace(case.period[1], 0, res_y)
        if not np.allclose(reference['y'], y, rtol=0, atol=case.period[1] * 1e-12):
            raise ValueError('RETICOLO y samples do not match the Meent y grid')
    elif reference['y'].size != 1:
        raise ValueError(
            f'expected one y sample for a 1D case, got {reference["y"].size}')

    # RETICOLO measures z upward from the bottom of the stack and the loader flips the
    # field arrays, so the reversed grid is a depth measured downward from the top.
    total = float(np.sum(thickness))
    depth = total - reference['z_raw'][::-1]
    if not np.allclose(_z_coordinates(thickness, res_z), depth, rtol=0, atol=total * 1e-12):
        raise ValueError('RETICOLO z samples do not match the Meent per-layer z grid')


def edge_entries(case):
    """(region, component) pairs that straddle a material edge.

    The grating is the only laterally structured layer, so it holds the only material
    edge. The electric component normal to that edge is genuinely discontinuous, and both
    solvers reconstruct it from a truncated Fourier series, so they disagree in the Gibbs
    region around the edge even when everything else agrees. Those entries get their own
    tolerance rather than being dropped from the table.
    """
    normals = ('Ex',) if case.dim == 1 else ('Ex', 'Ey')
    return {('grating', component) for component in normals}


def field_error_table(reference, candidate, res_z):
    rows = []
    for pol, pol_name in enumerate(POL_NAMES):
        for region, zslice in zip(REGIONS, _region_slices(res_z, len(REGIONS))):
            region_norm = np.linalg.norm(reference[pol, zslice])
            for component, name in enumerate(COMPONENTS):
                r = reference[pol, zslice, ..., component]
                m = candidate[pol, zslice, ..., component]
                denom = np.linalg.norm(r)
                # A symmetry-forbidden component can be numerical noise in both solvers.
                # Dividing by that near-zero component norm turns harmless roundoff into O(1).
                if denom < region_norm * 1e-8:
                    denom = region_norm
                denom = max(denom, np.finfo(float).eps)
                rel = np.linalg.norm(m - r) / denom
                rows.append((pol_name, region, name, float(rel)))
    return rows


def print_error_table(rows):
    print(f"{'pol':4s} {'region':12s} " + ' '.join(f'{c:>10s}' for c in COMPONENTS))
    print('-' * 86)
    for pol in POL_NAMES:
        for region in REGIONS:
            values = {c: v for p, r, c, v in rows if p == pol and r == region}
            print(f'{pol:4s} {region:12s} ' + ' '.join(f'{values[c]:10.3e}' for c in COMPONENTS))


def _raw_parts(field):
    """(label, real 2D array) for each raw complex component, Re and Im separately.

    Not |Ex|^2 and friends. Squaring discards sign and phase, which is exactly where a
    convention error lives: a field off by -1, by a conjugation, or by a global phase has
    a pixel-identical intensity map. Re and Im together are the complex value itself, so
    nothing is hidden and nothing is derived.

    This is also what ``field_error_table`` already measures - it subtracts the complex
    arrays directly - so the plots now show the same quantity the verdict is computed
    from. |E|^2 and |H|^2 are dropped for the same reason: they are functions of these
    six components, so they cannot disagree if these agree.
    """
    for i, name in enumerate(COMPONENTS):
        yield f'Re {name}', field[..., i].real
        yield f'Im {name}', field[..., i].imag


def _plot_plane(reference, candidate, coords_h, coords_v, plane_name, title, save_path=None):
    parts_r = list(_raw_parts(reference))
    parts_m = list(_raw_parts(candidate))
    fig, axes = plt.subplots(len(parts_r), 3, figsize=(13, 2.7 * len(parts_r)),
                             squeeze=False)
    extent = [coords_h[0], coords_h[-1], coords_v[-1], coords_v[0]]
    for row, ((label, r), (_, m)) in enumerate(zip(parts_r, parts_m)):
        # One symmetric scale shared by the RETICOLO and Meent panels. Signed data on a
        # diverging map centered at zero, so a sign flip reads as an inverted colour
        # rather than as an identical picture.
        scale = max(float(np.nanmax(np.abs(r))), float(np.nanmax(np.abs(m))),
                    np.finfo(float).eps)
        panels = (
            (r, f'RETICOLO {label}', 'RdBu_r', -scale, scale),
            (m, f'Meent {label}', 'RdBu_r', -scale, scale),
            (np.abs(m - r) / scale, f'|difference| / max|{label}|', 'magma', 0.0, None),
        )
        for col, (image, text, cmap, vmin, vmax) in enumerate(panels):
            im = axes[row, col].imshow(image, extent=extent, origin='upper', aspect='auto',
                                       cmap=cmap, vmin=vmin, vmax=vmax)
            axes[row, col].set_title(text)
            axes[row, col].set_xlabel(f'{plane_name[0]} (m)')
            axes[row, col].set_ylabel(f'{plane_name[1]} (m)')
            fig.colorbar(im, ax=axes[row, col], shrink=.8)
    fig.suptitle(title, y=.999)
    fig.tight_layout(rect=(0, 0, 1, .99))
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches='tight')
    return fig


def plot_field_profiles(result, save_plots=False):
    workdir = result['workdir']
    tag = wavelength_tag(result['wavelength_m'])
    method = result['method']
    figures = []
    for pol, pol_name in enumerate(POL_NAMES):
        r, m = result['reticolo'][pol], result['meent'][pol]
        ymid, xmid = r.shape[1] // 2, r.shape[2] // 2
        path = workdir / f'field_{tag}m_{method}_{pol_name}_xz.png' if save_plots else None
        figures.append(_plot_plane(
            r[:, ymid, :, :], m[:, ymid, :, :], result['x'], result['z'], 'xz',
            f'{result["case"].name}, {pol_name}, Meent {method}, '
            f'{result["wavelength_m"]:.6g} m, x-z at y center', path))
        if result['case'].dim == 2:
            path = workdir / f'field_{tag}m_{method}_{pol_name}_yz.png' if save_plots else None
            figures.append(_plot_plane(
                r[:, :, xmid, :], m[:, :, xmid, :], result['y'], result['z'], 'yz',
                f'{result["case"].name}, {pol_name}, Meent {method}, '
                f'{result["wavelength_m"]:.6g} m, y-z at x center', path))

            for layer, zslice in zip(REGIONS, _region_slices(result['res_z'], len(REGIONS))):
                zi = (zslice.start + zslice.stop - 1) // 2
                path = (workdir / f'field_{tag}m_{method}_{pol_name}_xy_{layer}.png'
                        if save_plots else None)
                figures.append(_plot_plane(
                    r[zi], m[zi], result['x'], result['y'], 'xy',
                    f'{result["case"].name}, {pol_name}, Meent {method}, {layer} center, '
                    f'{result["wavelength_m"]:.6g} m', path))
    return figures


def field_verdict(case, result):
    """Collect every field check and raise. A run that ends without an exception is the pass.

    The mirror of ``_compare.verdict`` for the field path. Everything above only prints, so
    without this a field regression would leave a notebook that still looks green.
    """
    method = result['method']
    tol, tol_edge = case.tol_field[method], case.tol_field_edge[method]
    relaxed = edge_entries(case)
    failures = []

    # Coverage first: an error table that lost entries would otherwise pass every
    # tolerance below by having nothing left to check.
    expected = {(p, r, c) for p in POL_NAMES for r in REGIONS for c in COMPONENTS}
    got = {(p, r, c) for p, r, c, _ in result['errors']}
    if got != expected:
        failures.append(f'field error table covers {len(got)} of {len(expected)} '
                        f'(pol, region, component) entries')

    for pol, region, component, value in result['errors']:
        if not np.isfinite(value):
            failures.append(f'field {pol} {region} {component}: not finite')
            continue
        limit = tol_edge if (region, component) in relaxed else tol
        if value > limit:
            failures.append(
                f'field {pol} {region} {component}: {value:.3e} > {limit:g}')

    print(f'{len(failures)} field failure(s)')
    for f in failures:
        print('  -', f)
    assert not failures, f'{len(failures)} field validation failure(s) - see above'
    print()
    print(f'PASS - Meent agrees with RETICOLO everywhere within {tol:g} '
          f'({tol_edge:g} for the electric component normal to the grating edge).')


def compare_field_profile(case, workdir, wavelength_m=600e-9, method='continuous',
                          save_plots=False, show_plots=True, make_plots=True,
                          raise_on_failure=True):
    """Load RETICOLO field, solve Meent once per polarization, compare, plot, and judge.

    Plots are produced before the verdict runs, so a failing run still leaves the figures
    that show where it failed. Pass ``raise_on_failure=False`` to get the result dict back
    from a run that would otherwise raise.
    """
    if case.evanescent:
        raise NotImplementedError('RETICOLO res3 does not expose the evanescent incident channel')
    workdir = Path(workdir)
    reference = load_reticolo_fields(case, workdir, wavelength_m)
    heights = reference['profile_heights']
    expected_layers = len(case.thickness) + 2
    if heights.size != expected_layers:
        raise ValueError(f'RETICOLO field has {heights.size} layers, expected {expected_layers}')
    if not np.allclose(heights[1:-1], case.thickness, rtol=0, atol=1e-15):
        raise ValueError('RETICOLO internal layer thicknesses do not match the Meent case')
    check_grid(case, reference, heights)

    meent_field = calculate_meent_fields(
        case, wavelength_m, method,
        reference['res_x'], reference['res_y'], reference['res_z'],
        heights[0], heights[-1],
    )
    reticolo_field = reference['field']
    if reticolo_field.shape != meent_field.shape:
        raise ValueError(
            f'field-grid mismatch: RETICOLO {reticolo_field.shape}, Meent {meent_field.shape}')

    z = _z_coordinates(heights, reference['res_z'])
    # Direct comparison: RETICOLO is untouched and Meent is already expressed in its
    # convention. No global phase or amplitude is fitted.
    rows = field_error_table(reticolo_field, meent_field, reference['res_z'])
    print(f'RETICOLO reference: {reference["path"].name}')
    print(f'Meent method: {method}; grid: z={len(z)}, y={len(reference["y"])}, '
          f'x={len(reference["x"])}')
    print('comparison: direct complex field (no phase/amplitude fit)')
    print_error_table(rows)

    result = {
        'case': case, 'workdir': workdir, 'wavelength_m': float(wavelength_m),
        'method': method, 'reticolo': reticolo_field, 'meent': meent_field,
        'errors': rows,
        'x': reference['x'], 'y': reference['y'], 'z': z,
        'res_z': reference['res_z'], 'profile_heights': heights,
    }
    result['figures'] = (plot_field_profiles(result, save_plots=save_plots)
                         if make_plots else [])
    if show_plots and make_plots:
        plt.show()
    if raise_on_failure:
        field_verdict(case, result)
    return result
