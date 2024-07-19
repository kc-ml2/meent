import numpy as np
from .fourier_analysis import dfs2d, cfs2d


def cell_compression(cell, type_complex=np.complex128):

    if type_complex == np.complex128:
        type_float = np.float64
    else:
        type_float = np.float32

    # find discontinuities in x
    step_y, step_x = 1. / np.array(cell.shape, dtype=type_float)
    x = []
    y = []
    cell_x = []
    cell_xy = []

    cell_next = np.roll(cell, -1, axis=1)

    for col in range(cell.shape[1]):
        if not (cell[:, col] == cell_next[:, col]).all() or (col == cell.shape[1] - 1):
            x.append(step_x * (col + 1))
            cell_x.append(cell[:, col])

    cell_x = np.array(cell_x).T
    cell_x_next = np.roll(cell_x, -1, axis=0)

    for row in range(cell_x.shape[0]):
        if not (cell_x[row, :] == cell_x_next[row, :]).all() or (row == cell_x.shape[0] - 1):
            y.append(step_y * (row + 1))
            cell_xy.append(cell_x[row, :])

    x = np.array(x).reshape((-1, 1))
    y = np.array(y).reshape((-1, 1))
    cell_comp = np.array(cell_xy)

    return cell_comp, x, y


def fft_piecewise_constant(cell, x, y, fourier_order_x, fourier_order_y, type_complex=np.complex128):

    period_x, period_y = x[-1], y[-1]

    # X axis
    cell_next_x = np.roll(cell, -1, axis=1)
    cell_diff_x = cell_next_x - cell
    cell_diff_x = cell_diff_x.astype(type_complex)

    cell = cell.astype(type_complex)

    modes_x = np.arange(-2 * fourier_order_x, 2 * fourier_order_x + 1, 1)

    f_coeffs_x = cell_diff_x @ np.exp(-1j * 2 * np.pi * x @ modes_x[None, :] / period_x, dtype=type_complex)
    c = f_coeffs_x.shape[1] // 2

    x_next = np.vstack((np.roll(x, -1, axis=0)[:-1], period_x)) - x

    f_coeffs_x[:, c] = (cell @ np.vstack((x[0], x_next[:-1]))).flatten() / period_x
    mask = np.ones(f_coeffs_x.shape[1], dtype=bool)
    mask[c] = False
    f_coeffs_x[:, mask] /= (1j * 2 * np.pi * modes_x[mask])

    # Y axis
    f_coeffs_x_next_y = np.roll(f_coeffs_x, -1, axis=0)
    f_coeffs_x_diff_y = f_coeffs_x_next_y - f_coeffs_x

    modes_y = np.arange(-2 * fourier_order_y, 2 * fourier_order_y + 1, 1)

    f_coeffs_xy = f_coeffs_x_diff_y.T @ np.exp(-1j * 2 * np.pi * y @ modes_y[None, :] / period_y, dtype=type_complex)
    c = f_coeffs_xy.shape[1] // 2

    y_next = np.vstack((np.roll(y, -1, axis=0)[:-1], period_y)) - y

    f_coeffs_xy[:, c] = f_coeffs_x.T @ np.vstack((y[0], y_next[:-1])).flatten() / period_y

    if c:
        mask = np.ones(f_coeffs_xy.shape[1], dtype=bool)
        mask[c] = False
        f_coeffs_xy[:, mask] /= (1j * 2 * np.pi * modes_y[mask])

    return f_coeffs_xy.T


def to_conv_mat_vector(ucell_info_list, fto_x, fto_y, device=None,
                       type_complex=np.complex128):

    ff_x = 2 * fto_x + 1
    ff_y = 2 * fto_y + 1

    epx_conv_all = np.zeros((len(ucell_info_list), ff_x * ff_y, ff_x * ff_y)).astype(type_complex)
    epy_conv_all = np.zeros((len(ucell_info_list), ff_x * ff_y, ff_x * ff_y)).astype(type_complex)
    epz_i_conv_all = np.zeros((len(ucell_info_list), ff_x * ff_y, ff_x * ff_y)).astype(type_complex)

    # 2D
    for i, ucell_info in enumerate(ucell_info_list):
        ucell_layer, x_list, y_list = ucell_info
        # ucell_layer = ucell_layer ** 2
        eps_compressed = ucell_layer ** 2

        epx_f = cfs2d(eps_compressed, x_list, y_list, fto_x, fto_y, 0, 1, type_complex)
        epy_f = cfs2d(eps_compressed, x_list, y_list, fto_x, fto_y, 1, 0, type_complex)
        epz_f = cfs2d(eps_compressed, x_list, y_list, fto_x, fto_y, 1, 1, type_complex)

        # center = np.array(f_coeffs.shape) // 2
        center = np.array(epz_f.shape) // 2

        conv_y = np.arange(-ff_y + 1, ff_y, 1)
        conv_y = circulant(conv_y)
        conv_y = np.repeat(conv_y, ff_x, axis=1)
        conv_y = np.repeat(conv_y, [ff_x] * ff_y, axis=0)

        conv_x = np.arange(-ff_x + 1, ff_x, 1)
        conv_x = circulant(conv_x)
        conv_x = np.tile(conv_x, (ff_y, ff_y))

        # e_conv = f_coeffs[center[0] + conv_i, center[1] + conv_j]
        # o_e_conv = o_f_coeffs[center[0] + conv_i, center[1] + conv_j]
        # e_conv_all[i] = e_conv
        # o_e_conv_all[i] = o_e_conv

        # XY to RC
        epx_conv = epx_f[center[0] + conv_y, center[1] + conv_x]
        epy_conv = epy_f[center[0] + conv_y, center[1] + conv_x]
        epz_conv = epz_f[center[0] + conv_y, center[1] + conv_x]

        epx_conv_all[i] = epx_conv
        epy_conv_all[i] = epy_conv
        epz_i_conv_all[i] = np.linalg.inv(epz_conv)

    # return e_conv_all, o_e_conv_all
    return epx_conv_all, epy_conv_all, epz_i_conv_all


def to_conv_mat_raster_continuous(ucell, fto_x, fto_y, device=None, type_complex=np.complex128):
    ucell_pmt = ucell ** 2

    if ucell_pmt.shape[1] == 1:  # 1D
        ff_x = 2 * fto_x + 1
        ff_y = 2 * fto_y + 1  # which is 1

        epx_conv_all = np.zeros((ucell_pmt.shape[0], ff_y * ff_x, ff_y * ff_x)).astype(type_complex)
        epy_conv_all = np.zeros((ucell_pmt.shape[0], ff_y * ff_x, ff_y * ff_x)).astype(type_complex)
        epz_conv_i_all = np.zeros((ucell_pmt.shape[0], ff_y * ff_x, ff_y * ff_x)).astype(type_complex)

        for i, layer in enumerate(ucell_pmt):

            eps_compressed, x, y = cell_compression(layer, type_complex=type_complex)

            epz_conv = cfs2d(eps_compressed, x, y, 1, 1, fto_x, fto_y, type_complex)
            epy_conv = cfs2d(eps_compressed, x, y, 1, 0, fto_x, fto_y, type_complex)
            epx_conv = cfs2d(eps_compressed, x, y, 0, 1, fto_x, fto_y, type_complex)

            # # center = np.array(f_coeffs.shape) // 2
            # center = np.array(epz_f.shape) // 2
            #
            # conv_x = np.arange(-ff + 1, ff, 1, dtype=int)
            # conv_x = circulant(conv_x)
            #
            # # XY to RC
            # epx_conv = epx_f[center[0], center[1] + conv_x]
            # epy_conv = epy_f[center[0], center[1] + conv_x]
            # epz_conv = epz_f[center[0], center[1] + conv_x]

            epx_conv_all[i] = epx_conv
            epy_conv_all[i] = epy_conv
            epz_conv_i_all[i] = np.linalg.inv(epz_conv)

    else:  # 2D
        ff_x = 2 * fto_x + 1
        ff_y = 2 * fto_y + 1

        epx_conv_all = np.zeros((ucell_pmt.shape[0], ff_y * ff_x, ff_y * ff_x)).astype(type_complex)
        epy_conv_all = np.zeros((ucell_pmt.shape[0], ff_y * ff_x, ff_y * ff_x)).astype(type_complex)
        epz_conv_i_all = np.zeros((ucell_pmt.shape[0], ff_y * ff_x, ff_y * ff_x)).astype(type_complex)

        for i, layer in enumerate(ucell_pmt):

            eps_compressed, x, y = cell_compression(layer, type_complex=type_complex)

            epz_conv = cfs2d(eps_compressed, x, y, 1, 1, fto_x, fto_y, type_complex)
            epy_conv = cfs2d(eps_compressed, x, y, 1, 0, fto_x, fto_y, type_complex)
            epx_conv = cfs2d(eps_compressed, x, y, 0, 1, fto_x, fto_y, type_complex)

            epx_conv_all[i] = epx_conv
            epy_conv_all[i] = epy_conv
            epz_conv_i_all[i] = np.linalg.inv(epz_conv)

    return epx_conv_all, epy_conv_all, epz_conv_i_all


def to_conv_mat_raster_discrete(ucell, fto_x, fto_y, device=None, type_complex=np.complex128,
                                enhanced_dfs=True):
    ucell_pmt = ucell ** 2

    if ucell_pmt.shape[1] == 1:  # 1D
        ff_x = 2 * fto_x + 1
        ff_y = 2 * fto_y + 1  # which is 1

        epx_conv_all = np.zeros((ucell_pmt.shape[0], ff_y * ff_x, ff_y * ff_x)).astype(type_complex)
        epy_conv_all = np.zeros((ucell_pmt.shape[0], ff_y * ff_x, ff_y * ff_x)).astype(type_complex)
        epz_conv_i_all = np.zeros((ucell_pmt.shape[0], ff_y * ff_x, ff_y * ff_x)).astype(type_complex)

        if enhanced_dfs:
            minimum_pattern_size_x = (4 * fto_x + 1) * ucell_pmt.shape[2]
        else:
            minimum_pattern_size_x = (4 * fto_x + 1)  # TODO: align with other bds

        for i, layer in enumerate(ucell_pmt):
            n = minimum_pattern_size_x // layer.shape[1]
            layer = np.repeat(layer, n + 1, axis=1)

            epz_conv = dfs2d(layer, 1, 1, fto_x, fto_y, type_complex)
            epy_conv = dfs2d(layer, 1, 0, fto_x, fto_y, type_complex)
            epx_conv = dfs2d(layer, 0, 1, fto_x, fto_y, type_complex)

            # # center = np.array(f_coeffs.shape) // 2
            # center = np.array(epz_f.shape) // 2
            #
            # conv_x = np.arange(-ff + 1, ff, 1, dtype=int)
            # conv_x = circulant(conv_x)
            #
            # # e_conv = f_coeffs[center[0], center[1] + conv_idx]
            # # o_e_conv = o_f_coeffs[center[0], center[1] + conv_idx]
            # # e_conv_all[i] = e_conv
            # # o_e_conv_all[i] = o_e_conv
            #
            # # XY to RC
            # epx_conv = epx_f[center[0], center[1] + conv_x]
            # epy_conv = epy_f[center[0], center[1] + conv_x]
            # epz_conv = epz_f[center[0], center[1] + conv_x]

            epx_conv_all[i] = epx_conv
            epy_conv_all[i] = epy_conv
            epz_conv_i_all[i] = np.linalg.inv(epz_conv)

    else:  # 2D
        ff_x = 2 * fto_x + 1
        ff_y = 2 * fto_y + 1

        epx_conv_all = np.zeros((ucell_pmt.shape[0], ff_y * ff_x, ff_y * ff_x)).astype(type_complex)
        epy_conv_all = np.zeros((ucell_pmt.shape[0], ff_y * ff_x, ff_y * ff_x)).astype(type_complex)
        epz_conv_i_all = np.zeros((ucell_pmt.shape[0], ff_y * ff_x, ff_y * ff_x)).astype(type_complex)

        if enhanced_dfs:
            minimum_pattern_size_y = (4 * fto_y + 1) * ucell_pmt.shape[1]
            minimum_pattern_size_x = (4 * fto_x + 1) * ucell_pmt.shape[2]
        else:
            minimum_pattern_size_y = 4 * fto_y + 1
            minimum_pattern_size_x = 4 * fto_x + 1
            # e.g., 8 bytes * (40*500) * (40*500) / 1E6 = 3200 MB = 3.2 GB

        for i, layer in enumerate(ucell_pmt):
            if layer.shape[0] < minimum_pattern_size_y:
                n = minimum_pattern_size_y // layer.shape[0]
                layer = np.repeat(layer, n + 1, axis=0)

            if layer.shape[1] < minimum_pattern_size_x:
                n = minimum_pattern_size_x // layer.shape[1]
                layer = np.repeat(layer, n + 1, axis=1)

            epz_conv = dfs2d(layer, 1, 1, fto_x, fto_y, type_complex)
            epy_conv = dfs2d(layer, 1, 0, fto_x, fto_y, type_complex)
            epx_conv = dfs2d(layer, 0, 1, fto_x, fto_y, type_complex)

            epx_conv_all[i] = epx_conv
            epy_conv_all[i] = epy_conv
            epz_conv_i_all[i] = np.linalg.inv(epz_conv)
            # a = np.linalg.inv(epz_conv)
            # epz_i_conv_all[i] = a[0][0]

    return epx_conv_all, epy_conv_all, epz_conv_i_all


def circulant(c):

    center = c.shape[0] // 2
    circ = np.zeros((center + 1, center + 1), dtype=int)

    for r in range(center+1):
        idx = np.arange(r, r - center - 1, -1, dtype=int)

        assign_value = c[center - idx]
        circ[r] = assign_value

    return circ
