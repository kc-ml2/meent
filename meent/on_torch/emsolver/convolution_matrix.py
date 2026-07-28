import torch
from .fourier_analysis import dfs2d, cfs2d
from .primitives import meeinv


def cell_compression(cell, device=torch.device('cpu'), type_complex=torch.complex128):

    cell = torch.flipud(cell)
    # This is needed because the comp. connecting_algo begins from 0 to period (RC coord. system).
    # On the other hand, the field data is from period to 0 (XY coord. system).
    # Will be flipped again during field reconstruction.

    if type_complex == torch.complex128:
        type_float = torch.float64
    else:
        type_float = torch.float32

    # find discontinuities in x
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


# TODO: delete
def fft_piecewise_constant(cell, x, y, fourier_order_x, fourier_order_y, device=torch.device('cpu'),
                           type_complex=torch.complex128):

    period_x, period_y = x[-1], y[-1]

    # X axis
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

    # Y axis
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

        if ucell_layer.ndim == 2:  # Isotropic: single refractive index per cell
            eps_x = eps_y = eps_z = ucell_layer ** 2
        elif ucell_layer.ndim == 3 and ucell_layer.shape[-1] == 3:  # Anisotropic: (nx, ny, nz)
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
        if layer.ndim == 2:  # Isotropic: single refractive index per cell
            lx = ly = lz = layer
        elif layer.ndim == 3 and layer.shape[-1] == 3:  # Anisotropic: (nx, ny, nz)
            lx, ly, lz = layer[..., 0], layer[..., 1], layer[..., 2]
        else:
            raise ValueError("layer must be 2D (isotropic) or 3D with 3 components (anisotropic)")

        # Share one compression only when the three components are the *same tensor*
        # (the isotropic branch above), never when they merely hold equal values.
        # Dispatching on value equality silently breaks the backward pass: the shared
        # path reads lx alone, so at a point where eps_x == eps_y == eps_z the gradient
        # of eps_y and eps_z comes back as exactly 0 while eps_x collects their sum.
        # The forward result is identical either way, so the bug is invisible until you
        # differentiate -- e.g. an optimization started from an isotropic material takes
        # one badly wrong step, which later iterations never undo.
        if lx is ly is lz:
            # compress once and share the discontinuity grid across components
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

    if enhanced_dfs:
        minimum_pattern_size_y = (4 * fto_y + 1) * ucell.shape[1]
        minimum_pattern_size_x = (4 * fto_x + 1) * ucell.shape[2]
    else:
        minimum_pattern_size_y = 4 * fto_y + 1
        minimum_pattern_size_x = 4 * fto_x + 1
        # e.g., 8 bytes * (40*500) * (40*500) / 1E6 = 3200 MB = 3.2 GB

    # ucell shape: (Layers, H, W) isotropic or (Layers, H, W, 3) anisotropic
    def _repeat_to_min(a):
        # Repeat the pattern up to the minimum size the requested Fourier order needs.
        if a.shape[0] < minimum_pattern_size_y:
            a = a.repeat_interleave((minimum_pattern_size_y // a.shape[0]) + 1, dim=0)
        if a.shape[1] < minimum_pattern_size_x:
            a = a.repeat_interleave((minimum_pattern_size_x // a.shape[1]) + 1, dim=1)
        return a

    for i, layer in enumerate(ucell):

        if layer.ndim == 2:  # Isotropic: nx=ny=nz, so repeat and square ONCE and share the
            # same eps across epx/epy/epz (they still need 3 separate DFTs with different
            # factorization rules, but the pattern array is not stored/repeated 3 times).
            eps_x = eps_y = eps_z = _repeat_to_min(layer) ** 2
        elif layer.ndim == 3 and layer.shape[-1] == 3:  # Anisotropic: (nx, ny, nz)
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
