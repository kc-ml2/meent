import numpy as np


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


def fft_piecewise_constant(cell,  fourier_order_x, fourier_order_y, type_complex=np.complex128):
    """
    reference: reticolo
    """

    cell, x, y = cell_compression(cell, type_complex=type_complex)

    # X axis
    cell_next_x = np.roll(cell, -1, axis=1)
    cell_diff_x = cell_next_x - cell

    modes_x = np.arange(-2 * fourier_order_x, 2 * fourier_order_x + 1, 1)

    f_coeffs_x = cell_diff_x @ np.exp(-1j * 2 * np.pi * x @ modes_x[None, :], dtype=type_complex)
    c = f_coeffs_x.shape[1] // 2

    # x_next = np.vstack(np.roll(x, -1, axis=0)[:-1]) - x
    x_next = np.vstack((np.roll(x, -1, axis=0)[:-1], 1)) - x

    f_coeffs_x[:, c] = (cell @ np.vstack((x[0], x_next[:-1]))).flatten()
    mask = np.ones(f_coeffs_x.shape[1], dtype=bool)
    mask[c] = False
    f_coeffs_x[:, mask] /= (1j * 2 * np.pi * modes_x[mask])

    # Y axis
    f_coeffs_x_next_y = np.roll(f_coeffs_x, -1, axis=0)
    f_coeffs_x_diff_y = f_coeffs_x_next_y - f_coeffs_x

    modes_y = np.arange(-2 * fourier_order_y, 2 * fourier_order_y + 1, 1)

    f_coeffs_xy = f_coeffs_x_diff_y.T @ np.exp(-1j * 2 * np.pi * y @ modes_y[None, :], dtype=type_complex)
    c = f_coeffs_xy.shape[1] // 2

    y_next = np.vstack((np.roll(y, -1, axis=0)[:-1], 1)) - y

    f_coeffs_xy[:, c] = f_coeffs_x.T @ np.vstack((y[0], y_next[:-1])).flatten()

    if c:
        mask = np.ones(f_coeffs_xy.shape[1], dtype=bool)
        mask[c] = False
        f_coeffs_xy[:, mask] /= (1j * 2 * np.pi * modes_y[mask])

    return f_coeffs_xy.T


def fft_piecewise_constant_vector(cell, x, y, fourier_order_x, fourier_order_y, type_complex=np.complex128):
    # X axis
    cell_next_x = np.roll(cell, -1, axis=1)
    cell_diff_x = cell_next_x - cell

    modes_x = np.arange(-2 * fourier_order_x, 2 * fourier_order_x + 1, 1)

    f_coeffs_x = cell_diff_x @ np.exp(-1j * 2 * np.pi * x @ modes_x[None, :], dtype=type_complex)
    c = f_coeffs_x.shape[1] // 2

    x_next = np.vstack((np.roll(x, -1, axis=0)[:-1], 1)) - x

    f_coeffs_x[:, c] = (cell @ np.vstack((x[0], x_next[:-1]))).flatten()
    mask = np.ones(f_coeffs_x.shape[1], dtype=bool)
    mask[c] = False
    f_coeffs_x[:, mask] /= (1j * 2 * np.pi * modes_x[mask])

    # Y axis
    f_coeffs_x_next_y = np.roll(f_coeffs_x, -1, axis=0)
    f_coeffs_x_diff_y = f_coeffs_x_next_y - f_coeffs_x

    modes_y = np.arange(-2 * fourier_order_y, 2 * fourier_order_y + 1, 1)

    f_coeffs_xy = f_coeffs_x_diff_y.T @ np.exp(-1j * 2 * np.pi * y @ modes_y[None, :], dtype=type_complex)
    c = f_coeffs_xy.shape[1] // 2

    y_next = np.vstack((np.roll(y, -1, axis=0)[:-1], 1)) - y

    f_coeffs_xy[:, c] = f_coeffs_x.T @ np.vstack((y[0], y_next[:-1])).flatten()

    if c:
        mask = np.ones(f_coeffs_xy.shape[1], dtype=bool)
        mask[c] = False
        f_coeffs_xy[:, mask] /= (1j * 2 * np.pi * modes_y[mask])

    return f_coeffs_xy.T


def to_conv_mat_continuous_vector(ucell_info_list, fourier_order_x, fourier_order_y, device=None,
                                  type_complex=np.complex128):
    ff_x = 2 * fourier_order_x + 1
    ff_y = 2 * fourier_order_y + 1

    e_conv_all = np.zeros((len(ucell_info_list), ff_x * ff_y, ff_x * ff_y)).astype(type_complex)
    o_e_conv_all = np.zeros((len(ucell_info_list), ff_x * ff_y, ff_x * ff_y)).astype(type_complex)

    # 2D  # tODO: 1D
    for i, ucell_info in enumerate(ucell_info_list):
        ucell_layer, x_list, y_list = ucell_info
        ucell_layer = ucell_layer ** 2

        f_coeffs = fft_piecewise_constant_vector(ucell_layer, x_list, y_list,
                                                 fourier_order_x, fourier_order_y, type_complex=type_complex)
        o_f_coeffs = fft_piecewise_constant_vector(1/ucell_layer, x_list, y_list,
                                                 fourier_order_x, fourier_order_y, type_complex=type_complex)
        center = np.array(f_coeffs.shape) // 2

        conv_idx_y = np.arange(-ff_y + 1, ff_y, 1)
        conv_idx_y = circulant(conv_idx_y)
        conv_i = np.repeat(conv_idx_y, ff_x, axis=1)
        conv_i = np.repeat(conv_i, [ff_x] * ff_y, axis=0)

        conv_idx_x = np.arange(-ff_x + 1, ff_x, 1)
        conv_idx_x = circulant(conv_idx_x)
        conv_j = np.tile(conv_idx_x, (ff_y, ff_y))

        e_conv = f_coeffs[center[0] + conv_i, center[1] + conv_j]
        o_e_conv = o_f_coeffs[center[0] + conv_i, center[1] + conv_j]

        e_conv_all[i] = e_conv
        o_e_conv_all[i] = o_e_conv

    return e_conv_all, o_e_conv_all


def to_conv_mat_continuous(ucell, fourier_order_x, fourier_order_y, device=None, type_complex=np.complex128):
    ucell_pmt = ucell ** 2

    if ucell_pmt.shape[1] == 1:  # 1D
        ff = 2 * fourier_order_x + 1

        e_conv_all = np.zeros((ucell_pmt.shape[0], ff, ff)).astype(type_complex)
        o_e_conv_all = np.zeros((ucell_pmt.shape[0], ff, ff)).astype(type_complex)

        for i, layer in enumerate(ucell_pmt):
            f_coeffs = fft_piecewise_constant(layer, fourier_order_x, fourier_order_y, type_complex=type_complex)
            o_f_coeffs = fft_piecewise_constant(1/layer, fourier_order_x, fourier_order_y, type_complex=type_complex)
            center = np.array(f_coeffs.shape) // 2
            conv_idx = np.arange(-ff + 1, ff, 1, dtype=int)
            conv_idx = circulant(conv_idx)
            e_conv = f_coeffs[center[0], center[1] + conv_idx]
            o_e_conv = o_f_coeffs[center[0], center[1] + conv_idx]
            e_conv_all[i] = e_conv
            o_e_conv_all[i] = o_e_conv
    else:  # 2D
        ff_x = 2 * fourier_order_x + 1
        ff_y = 2 * fourier_order_y + 1

        e_conv_all = np.zeros((ucell_pmt.shape[0], ff_x * ff_y,  ff_x * ff_y)).astype(type_complex)
        o_e_conv_all = np.zeros((ucell_pmt.shape[0], ff_x * ff_y,  ff_x * ff_y)).astype(type_complex)

        for i, layer in enumerate(ucell_pmt):
            f_coeffs = fft_piecewise_constant(layer, fourier_order_x, fourier_order_y, type_complex=type_complex)
            o_f_coeffs = fft_piecewise_constant(1/layer, fourier_order_x, fourier_order_y, type_complex=type_complex)
            center = np.array(f_coeffs.shape) // 2

            conv_idx_y = np.arange(-ff_y + 1, ff_y, 1)
            conv_idx_y = circulant(conv_idx_y)
            conv_i = np.repeat(conv_idx_y, ff_x, axis=1)
            conv_i = np.repeat(conv_i, [ff_x] * ff_y, axis=0)

            conv_idx_x = np.arange(-ff_x + 1, ff_x, 1)
            conv_idx_x = circulant(conv_idx_x)
            conv_j = np.tile(conv_idx_x, (ff_y, ff_y))

            e_conv = f_coeffs[center[0] + conv_i, center[1] + conv_j]
            o_e_conv = o_f_coeffs[center[0] + conv_i, center[1] + conv_j]
            e_conv_all[i] = e_conv
            o_e_conv_all[i] = o_e_conv
    return e_conv_all, o_e_conv_all


def to_conv_mat_discrete(ucell, fourier_order_x, fourier_order_y, device=None, type_complex=np.complex128,
                         improve_dft=True):
    ucell_pmt = ucell ** 2

    if ucell_pmt.shape[1] == 1:  # 1D
        ff = 2 * fourier_order_x + 1
        e_conv_all = np.zeros((ucell_pmt.shape[0], ff, ff)).astype(type_complex)
        o_e_conv_all = np.zeros((ucell_pmt.shape[0], ff, ff)).astype(type_complex)
        if improve_dft:
            minimum_pattern_size = 2 * ff * ucell_pmt.shape[2]
        else:
            minimum_pattern_size = 2 * ff

        for i, layer in enumerate(ucell_pmt):
            n = minimum_pattern_size // layer.shape[1]
            layer = np.repeat(layer, n + 1, axis=1)
            f_coeffs = np.fft.fftshift(np.fft.fft(layer / layer.size))
            o_f_coeffs = np.fft.fftshift(np.fft.fft(1/layer / layer.size))
            # FFT scaling:
            # https://kr.mathworks.com/matlabcentral/answers/15770-scaling-the-fft-and-the-ifft?s_tid=srchtitle

            center = np.array(f_coeffs.shape) // 2

            conv_idx = np.arange(-ff + 1, ff, 1, dtype=int)
            conv_idx = circulant(conv_idx)
            e_conv = f_coeffs[center[0], center[1] + conv_idx]
            o_e_conv = o_f_coeffs[center[0], center[1] + conv_idx]
            e_conv_all[i] = e_conv
            o_e_conv_all[i] = o_e_conv
    else:  # 2D
        ff_x = 2 * fourier_order_x + 1
        ff_y = 2 * fourier_order_y + 1

        e_conv_all = np.zeros((ucell_pmt.shape[0], ff_x * ff_y, ff_x * ff_y)).astype(type_complex)
        o_e_conv_all = np.zeros((ucell_pmt.shape[0], ff_x * ff_y, ff_x * ff_y)).astype(type_complex)

        if improve_dft:
            minimum_pattern_size_y = 2 * ff_y * ucell_pmt.shape[1]
            minimum_pattern_size_x = 2 * ff_x * ucell_pmt.shape[2]
        else:
            minimum_pattern_size_y = 2 * ff_y
            minimum_pattern_size_x = 2 * ff_x
        # e.g., 8 bytes * (40*500) * (40*500) / 1E6 = 3200 MB = 3.2 GB

        for i, layer in enumerate(ucell_pmt):
            if layer.shape[0] < minimum_pattern_size_y:
                n = minimum_pattern_size_y // layer.shape[0]
                layer = np.repeat(layer, n + 1, axis=0)

            if layer.shape[1] < minimum_pattern_size_x:
                n = minimum_pattern_size_x // layer.shape[1]
                layer = np.repeat(layer, n + 1, axis=1)

            f_coeffs = np.fft.fftshift(np.fft.fft2(layer / layer.size))
            o_f_coeffs = np.fft.fftshift(np.fft.fft2(1/layer / layer.size))
            center = np.array(f_coeffs.shape) // 2

            conv_idx_y = np.arange(-ff_y + 1, ff_y, 1)
            conv_idx_y = circulant(conv_idx_y)
            conv_i = np.repeat(conv_idx_y, ff_x, axis=1)
            conv_i = np.repeat(conv_i, [ff_x] * ff_y, axis=0)

            conv_idx_x = np.arange(-ff_x + 1, ff_x, 1)
            conv_idx_x = circulant(conv_idx_x)
            conv_j = np.tile(conv_idx_x, (ff_y, ff_y))

            e_conv = f_coeffs[center[0] + conv_i, center[1] + conv_j]
            o_e_conv = o_f_coeffs[center[0] + conv_i, center[1] + conv_j]
            e_conv_all[i] = e_conv
            o_e_conv_all[i] = o_e_conv
    return e_conv_all, o_e_conv_all


def circulant(c):

    center = c.shape[0] // 2
    circ = np.zeros((center + 1, center + 1), dtype=int)

    for r in range(center+1):
        idx = np.arange(r, r - center - 1, -1, dtype=int)

        assign_value = c[center - idx]
        circ[r] = assign_value

    return circ
