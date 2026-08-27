"""Turning a layer's permittivity into the three convolution matrices the solver needs.

One entry point per way of describing a layer:

    to_conv_mat_vector             list of (values, x edges, y edges)  - modeling_type 1
    to_conv_mat_raster_continuous  raster, exact Fourier integral      - fourier_type 1
    to_conv_mat_raster_discrete    raster, FFT                         - fourier_type 0

All three return the same triple, and each applies a different factorization rule, because the
field component each one multiplies is continuous in a different direction:

    epz_conv    (1, 1)   Ez against a discontinuity in neither direction
    epy_conv    (1, 0)   inverse rule along y
    epx_conv    (0, 1)   inverse rule along x

epz is returned already inverted - the solver only ever uses it that way, and inverting once
here beats inverting per layer per solve.
"""

import torch
from .fourier_analysis import dfs2d, cfs2d
from .primitives import meeinv


def cell_compression(cell, device=torch.device('cpu'), type_complex=torch.complex128):
    """Collapse a raster layer to its distinct rows and columns, with their y and x edges.

    Row index is the y axis and both run in the increasing direction, `y = step_y*(row+1)`,
    which is the mapping the raster builders, `dfs2d` (fourier_type=0) and the vector model
    all use. A `torch.flipud(cell)` used to stand at the top of this function while the y
    coordinates below were still assigned in row order, so this path alone read the cell's y
    axis upside down and solved the mirror image of the requested structure. It cancelled on
    a cell with a `y = const` mirror line - which every 2D case in validation/simulation had
    until they were made asymmetric - and on 1D, where flipping a single row does nothing.
    Measured before removal, on an asymmetric cell: continuous(+phi) == flipY[vector(-phi)]
    to 9e-14, against a 0.11 disagreement with vector at the same phi.
    """
    if type_complex == torch.complex128:
        type_float = torch.float64
    else:
        type_float = torch.float32

    step_y, step_x = 1. / torch.tensor(cell.shape, device=device, dtype=type_float)
    x = []
    y = []
    cell_x = []
    cell_xy = []

    cell_next = torch.roll(cell, -1, dims=1)

    for col in range(cell.shape[1]):
        if not (cell[:, col] == cell_next[:, col]).all() or (col == cell.shape[1] - 1):
            x.append(step_x * (col + 1))
            cell_x.append(cell[:, col].reshape((1, -1)))

    cell_x = torch.cat(cell_x, dim=0).T
    cell_x_next = torch.roll(cell_x, -1, dims=0)

    for row in range(cell_x.shape[0]):
        if not (cell_x[row, :] == cell_x_next[row, :]).all() or (row == cell_x.shape[0] - 1):
            y.append(step_y * (row + 1))
            cell_xy.append(cell_x[row, :].reshape((1, -1)))

    x = torch.tensor(x, device=device).reshape((-1, 1)).type(type_complex)
    y = torch.tensor(y, device=device).reshape((-1, 1)).type(type_complex)
    cell_comp = torch.cat(cell_xy, dim=0)

    return cell_comp, x, y


def fft_piecewise_constant(cell, x, y, fourier_order_x, fourier_order_y, device=torch.device('cpu'),
                           type_complex=torch.complex128):
    """Superseded by `_cfs` / `cfs2d` in fourier_analysis.py, and called from nowhere.

    Same idea as `_cfs` - deltas at the jumps, divided by i*2*pi*m - but written out for both
    directions in one function and returning raw coefficients rather than a convolution matrix.
    Kept only as the earlier, more literal statement of that derivation.
    """
    period_x, period_y = x[-1], y[-1]

    cell_next_x = torch.roll(cell, -1, dims=1)
    cell_diff_x = cell_next_x - cell
    cell_diff_x = cell_diff_x.type(type_complex)

    cell = cell.type(type_complex)

    modes_x = torch.arange(-2 * fourier_order_x, 2 * fourier_order_x + 1, 1, device=device).type(type_complex)

    f_coeffs_x = cell_diff_x @ torch.exp(-1j * 2 * torch.pi * x @ modes_x[None, :] / period_x).type(type_complex)
    c = f_coeffs_x.shape[1] // 2

    x_next = torch.vstack((torch.roll(x, -1, dims=0)[:-1], torch.tensor([period_x], device=device))) - x

    f_coeffs_x[:, c] = (cell @ torch.vstack((x[0], x_next[:-1]))).flatten() / period_x
    mask = torch.ones(f_coeffs_x.shape[1], device=device).type(torch.bool)
    mask[c] = False
    f_coeffs_x[:, mask] /= (1j * 2 * torch.pi * modes_x[mask])

    f_coeffs_x_next_y = torch.roll(f_coeffs_x, -1, dims=0)
    f_coeffs_x_diff_y = f_coeffs_x_next_y - f_coeffs_x

    modes_y = torch.arange(-2 * fourier_order_y, 2 * fourier_order_y + 1, 1, device=device).type(type_complex)

    f_coeffs_xy = f_coeffs_x_diff_y.T @ torch.exp(-1j * 2 * torch.pi * y @ modes_y[None, :] / period_y)
    c = f_coeffs_xy.shape[1] // 2

    y_next = torch.vstack((torch.roll(y, -1, dims=0)[:-1], torch.tensor([period_y], device=device))) - y

    f_coeffs_xy[:, c] = f_coeffs_x.T @ torch.vstack((y[0], y_next[:-1])).flatten() / period_y

    if c:
        mask = torch.ones(f_coeffs_xy.shape[1], device=device).type(torch.bool)
        mask[c] = False
        f_coeffs_xy[:, mask] /= (1j * 2 * torch.pi * modes_y[mask])

    return f_coeffs_xy.T


def to_conv_mat_vector(ucell_info_list, fto_x, fto_y, device=torch.device('cpu'), type_complex=torch.complex128,
                       use_pinv=False):

    ff_xy = (2 * fto_x + 1) * (2 * fto_y + 1)

    epx_conv_all = torch.zeros((len(ucell_info_list), ff_xy, ff_xy), device=device).type(type_complex)
    epy_conv_all = torch.zeros((len(ucell_info_list), ff_xy, ff_xy), device=device).type(type_complex)
    epz_conv_i_all = torch.zeros((len(ucell_info_list), ff_xy, ff_xy), device=device).type(type_complex)

    for i, ucell_info in enumerate(ucell_info_list):
        ucell_layer, x_list, y_list = ucell_info

        # The stored quantity is a refractive index; the solver works in permittivity, hence
        # the square. An anisotropic layer carries (nx, ny, nz) in a trailing axis of 3, and
        # only the diagonal of the permittivity tensor is representable this way.
        if ucell_layer.ndim == 2:
            eps_x = eps_y = eps_z = ucell_layer ** 2
        elif ucell_layer.ndim == 3 and ucell_layer.shape[-1] == 3:
            eps_x = ucell_layer[..., 0] ** 2
            eps_y = ucell_layer[..., 1] ** 2
            eps_z = ucell_layer[..., 2] ** 2
        else:
            raise ValueError("ucell_layer must be 2D (isotropic) or 3D with 3 components (anisotropic)")

        epz_conv = cfs2d(eps_z, x_list, y_list, 1, 1, fto_x, fto_y, device=device, type_complex=type_complex)
        epy_conv = cfs2d(eps_y, x_list, y_list, 1, 0, fto_x, fto_y, device=device, type_complex=type_complex)
        epx_conv = cfs2d(eps_x, x_list, y_list, 0, 1, fto_x, fto_y, device=device, type_complex=type_complex)

        epx_conv_all[i] = epx_conv
        epy_conv_all[i] = epy_conv
        epz_conv_i_all[i] = meeinv(epz_conv, use_pinv=use_pinv)

    return epx_conv_all, epy_conv_all, epz_conv_i_all


def to_conv_mat_raster_continuous(ucell, fto_x, fto_y, device=torch.device('cpu'), type_complex=torch.complex128,
                                  use_pinv=False):
    ff_xy = (2 * fto_x + 1) * (2 * fto_y + 1)

    epx_conv_all = torch.zeros((ucell.shape[0], ff_xy, ff_xy), device=device, dtype=type_complex)
    epy_conv_all = torch.zeros((ucell.shape[0], ff_xy, ff_xy), device=device, dtype=type_complex)
    epz_conv_i_all = torch.zeros((ucell.shape[0], ff_xy, ff_xy), device=device, dtype=type_complex)

    for i, layer in enumerate(ucell):
        if layer.ndim == 2:
            lx = ly = lz = layer
        elif layer.ndim == 3 and layer.shape[-1] == 3:
            lx, ly, lz = layer[..., 0], layer[..., 1], layer[..., 2]
        else:
            raise ValueError("layer must be 2D (isotropic) or 3D with 3 components (anisotropic)")

        # `is`, not `==`. The three names point at one tensor only on the isotropic branch
        # above, and only then may one compression be shared: the shared path reads lx alone,
        # so if this were dispatched on equal *values* instead, an anisotropic layer that
        # happens to have eps_x == eps_y == eps_z at some point would send the gradient of
        # eps_y and eps_z to zero and hand their share to eps_x. The forward result is
        # identical either way, so it stays invisible until something differentiates through
        # it - an optimization started from an isotropic material takes one badly wrong step
        # that later iterations never undo.
        if lx is ly is lz:
            n_compressed, x_list, y_list = cell_compression(lx, device=device, type_complex=type_complex)
            eps_x = eps_y = eps_z = n_compressed ** 2
            x_x = x_y = x_z = x_list
            y_x = y_y = y_z = y_list
        else:
            nx_compressed, x_x, y_x = cell_compression(lx, device=device, type_complex=type_complex)
            ny_compressed, x_y, y_y = cell_compression(ly, device=device, type_complex=type_complex)
            nz_compressed, x_z, y_z = cell_compression(lz, device=device, type_complex=type_complex)
            eps_x = nx_compressed ** 2
            eps_y = ny_compressed ** 2
            eps_z = nz_compressed ** 2

        epz_conv = cfs2d(eps_z, x_z, y_z, 1, 1, fto_x, fto_y, device=device, type_complex=type_complex)
        epy_conv = cfs2d(eps_y, x_y, y_y, 1, 0, fto_x, fto_y, device=device, type_complex=type_complex)
        epx_conv = cfs2d(eps_x, x_x, y_x, 0, 1, fto_x, fto_y, device=device, type_complex=type_complex)

        epx_conv_all[i] = epx_conv
        epy_conv_all[i] = epy_conv
        epz_conv_i_all[i] = meeinv(epz_conv, use_pinv=use_pinv)

    return epx_conv_all, epy_conv_all, epz_conv_i_all


def to_conv_mat_raster_discrete(ucell, fto_x, fto_y, device=torch.device('cpu'), type_complex=torch.complex128,
                                enhanced_dfs=True, use_pinv=False):

    ff_xy = (2 * fto_x + 1) * (2 * fto_y + 1)

    epx_conv_all = torch.zeros((ucell.shape[0], ff_xy, ff_xy), device=device).type(type_complex)
    epy_conv_all = torch.zeros((ucell.shape[0], ff_xy, ff_xy), device=device).type(type_complex)
    epz_conv_i_all = torch.zeros((ucell.shape[0], ff_xy, ff_xy), device=device).type(type_complex)

    # An FFT cannot report an order the sampling does not resolve. The convolution matrix needs
    # orders out to +-2*fto, so the raster has to carry at least 4*fto + 1 samples per period
    # or the highest orders come back aliased. `enhanced_dfs` multiplies that floor by the
    # layer's own sample count, upsampling each existing cell rather than merely clearing the
    # bar - finer sampling of the same step edges, at a cost that grows as the square.
    if enhanced_dfs:
        minimum_pattern_size_y = (4 * fto_y + 1) * ucell.shape[1]
        minimum_pattern_size_x = (4 * fto_x + 1) * ucell.shape[2]
    else:
        minimum_pattern_size_y = 4 * fto_y + 1
        minimum_pattern_size_x = 4 * fto_x + 1

    def _repeat_to_min(a):
        # repeat_interleave, not tile: each cell is duplicated in place, so the structure is
        # resampled and not repeated. Tiling would shrink the period instead.
        if a.shape[0] < minimum_pattern_size_y:
            a = a.repeat_interleave((minimum_pattern_size_y // a.shape[0]) + 1, dim=0)
        if a.shape[1] < minimum_pattern_size_x:
            a = a.repeat_interleave((minimum_pattern_size_x // a.shape[1]) + 1, dim=1)
        return a

    for i, layer in enumerate(ucell):

        if layer.ndim == 2:
            eps_x = eps_y = eps_z = _repeat_to_min(layer) ** 2
        elif layer.ndim == 3 and layer.shape[-1] == 3:
            eps_x = _repeat_to_min(layer[..., 0]) ** 2
            eps_y = _repeat_to_min(layer[..., 1]) ** 2
            eps_z = _repeat_to_min(layer[..., 2]) ** 2
        else:
            raise ValueError("layer must be 2D (isotropic) or 3D with 3 components (anisotropic)")

        epz_conv = dfs2d(eps_z, 1, 1, fto_x, fto_y, device=device, type_complex=type_complex)
        epy_conv = dfs2d(eps_y, 1, 0, fto_x, fto_y, device=device, type_complex=type_complex)
        epx_conv = dfs2d(eps_x, 0, 1, fto_x, fto_y, device=device, type_complex=type_complex)

        epx_conv_all[i] = epx_conv
        epy_conv_all[i] = epy_conv
        epz_conv_i_all[i] = meeinv(epz_conv, use_pinv=use_pinv)

    return epx_conv_all, epy_conv_all, epz_conv_i_all
