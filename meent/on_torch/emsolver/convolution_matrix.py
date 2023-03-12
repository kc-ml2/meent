import torch
import numpy as np


def cell_compression(cell, device=torch.device('cpu'), type_complex=torch.complex128):

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


def fft_piecewise_constant(cell, fourier_order, device=torch.device('cpu'), type_complex=torch.complex128):
    if cell.shape[0] == 1:
        fourier_order = [0, fourier_order]
    else:
        fourier_order = [fourier_order, fourier_order]
    cell, x, y = cell_compression(cell, device=device, type_complex=type_complex)

    # X axis
    cell_next_x = torch.roll(cell, -1, dims=1)
    cell_diff_x = cell_next_x - cell

    modes = torch.arange(-2 * fourier_order[1], 2 * fourier_order[1] + 1, 1, device=device).type(type_complex)

    cell_diff_x = cell_diff_x.type(type_complex)
    f_coeffs_x = cell_diff_x @ torch.exp(-1j * 2 * np.pi * x @ modes[None, :]).type(type_complex)
    c = f_coeffs_x.shape[1] // 2

    cell = cell.type(type_complex)
    x_next = torch.vstack((torch.roll(x, -1, dims=0)[:-1], torch.tensor([1], device=device))) - x

    f_coeffs_x[:, c] = (cell @ torch.vstack((x[0], x_next[:-1]))).flatten()
    mask = torch.ones(f_coeffs_x.shape[1], device=device).type(torch.bool)
    mask[c] = False
    f_coeffs_x[:, mask] /= (1j * 2 * np.pi * modes[mask])

    # Y axis
    f_coeffs_x_next_y = torch.roll(f_coeffs_x, -1, dims=0)
    f_coeffs_x_diff_y = f_coeffs_x_next_y - f_coeffs_x

    modes = torch.arange(-2 * fourier_order[0], 2 * fourier_order[0] + 1, 1, device=device).type(type_complex)

    f_coeffs_xy = f_coeffs_x_diff_y.T @ torch.exp(-1j * 2 * np.pi * y @ modes[None, :])
    c = f_coeffs_xy.shape[1] // 2

    y_next = torch.vstack((torch.roll(y, -1, dims=0)[:-1], torch.tensor([1], device=device))) - y

    f_coeffs_xy[:, c] = f_coeffs_x.T @ torch.vstack((y[0], y_next[:-1])).flatten()

    if c:
        mask = torch.ones(f_coeffs_xy.shape[1], device=device).type(torch.bool)
        mask[c] = False
        f_coeffs_xy[:, mask] /= (1j * 2 * np.pi * modes[mask])

    return f_coeffs_xy.T


def fft_piecewise_constant_vector(cell, x, y, fourier_order, device=torch.device('cpu'), type_complex=torch.complex128):
    if cell.shape[0] == 1:
        fourier_order = [0, fourier_order]  # tODO
    else:
        fourier_order = [fourier_order, fourier_order]
    # cell, x, y = cell_compression(cell, device=device, type_complex=type_complex)

    # X axis
    cell_next_x = torch.roll(cell, -1, dims=1)
    cell_diff_x = cell_next_x - cell

    modes = torch.arange(-2 * fourier_order[1], 2 * fourier_order[1] + 1, 1, device=device).type(type_complex)

    cell_diff_x = cell_diff_x.type(type_complex)
    f_coeffs_x = cell_diff_x @ torch.exp(-1j * 2 * np.pi * x @ modes[None, :]).type(type_complex)
    c = f_coeffs_x.shape[1] // 2

    cell = cell.type(type_complex)
    x_next = torch.vstack((torch.roll(x, -1, dims=0)[:-1], torch.tensor([1], device=device))) - x

    f_coeffs_x[:, c] = (cell @ torch.vstack((x[0], x_next[:-1]))).flatten()
    mask = torch.ones(f_coeffs_x.shape[1], device=device).type(torch.bool)
    mask[c] = False
    f_coeffs_x[:, mask] /= (1j * 2 * np.pi * modes[mask])

    # Y axis
    f_coeffs_x_next_y = torch.roll(f_coeffs_x, -1, dims=0)
    f_coeffs_x_diff_y = f_coeffs_x_next_y - f_coeffs_x

    modes = torch.arange(-2 * fourier_order[0], 2 * fourier_order[0] + 1, 1, device=device).type(type_complex)

    f_coeffs_xy = f_coeffs_x_diff_y.T @ torch.exp(-1j * 2 * np.pi * y @ modes[None, :])
    c = f_coeffs_xy.shape[1] // 2

    y_next = torch.vstack((torch.roll(y, -1, dims=0)[:-1], torch.tensor([1], device=device))) - y

    f_coeffs_xy[:, c] = f_coeffs_x.T @ torch.vstack((y[0], y_next[:-1])).flatten()

    if c:
        mask = torch.ones(f_coeffs_xy.shape[1], device=device).type(torch.bool)
        mask[c] = False
        f_coeffs_xy[:, mask] /= (1j * 2 * np.pi * modes[mask])

    return f_coeffs_xy.T


def to_conv_mat_continuous_vector(ucell_info_list, fourier_order, device=torch.device('cpu'), type_complex=torch.complex128):

    ff = 2 * fourier_order + 1

    e_conv_all = torch.zeros((len(ucell_info_list), ff ** 2, ff ** 2)).type(type_complex)
    o_e_conv_all = torch.zeros((len(ucell_info_list), ff ** 2, ff ** 2)).type(type_complex)

    # 2D  # tODO: 1D
    for i, ucell_info in enumerate(ucell_info_list):
        ucell_layer, x_list, y_list = ucell_info
        # ucell_layer = torch.tensor(ucell_layer, dtype=type_complex) if type(ucell_layer) != torch.Tensor else ucell_layer
        # x_list = torch.tensor(x_list, dtype=type_complex) if type(x_list) != torch.Tensor else x_list
        # y_list = torch.tensor(y_list, dtype=type_complex) if type(y_list) != torch.Tensor else y_list

        f_coeffs = fft_piecewise_constant_vector(ucell_layer, x_list, y_list,
                                                 fourier_order, type_complex=type_complex)
        o_f_coeffs = fft_piecewise_constant_vector(1/ucell_layer, x_list, y_list,
                                                 fourier_order, type_complex=type_complex)

        center = torch.div(torch.tensor(f_coeffs.shape, device=device), 2, rounding_mode='trunc')

        conv_idx = torch.arange(-ff + 1, ff, 1, device=device).type(torch.long)
        conv_idx = circulant(conv_idx, device)
        conv_i = conv_idx.repeat_interleave(ff, dim=1).type(torch.long)
        conv_i = conv_i.repeat_interleave(ff, dim=0)
        conv_j = conv_idx.repeat(ff, ff).type(torch.long)

        e_conv = f_coeffs[center[0] + conv_i, center[1] + conv_j]
        o_e_conv = o_f_coeffs[center[0] + conv_i, center[1] + conv_j]

        e_conv_all[i] = e_conv
        o_e_conv_all[i] = o_e_conv

    return e_conv_all, o_e_conv_all


def to_conv_mat_continuous(pmt, fourier_order, device=torch.device('cpu'), type_complex=torch.complex128):

    if len(pmt.shape) == 2:
        print('shape is 2')
        raise ValueError

    # pmt = torch.tensor(pmt) if type(pmt) != torch.Tensor else pmt

    ff = 2 * fourier_order + 1

    if pmt.shape[1] == 1:  # 1D
        res = torch.zeros((pmt.shape[0], ff, ff), device=device).type(type_complex)

        for i, layer in enumerate(pmt):
            f_coeffs = fft_piecewise_constant(layer, fourier_order, device=device, type_complex=type_complex)
            center = f_coeffs.shape[1] // 2
            conv_idx = torch.arange(-ff + 1, ff, 1, device=device).type(torch.long)
            conv_idx = circulant(conv_idx, device)
            e_conv = f_coeffs[0, center + conv_idx]
            res[i] = e_conv

    else:  # 2D
        # attention on the order of axis (Z Y X)
        res = torch.zeros((pmt.shape[0], ff ** 2, ff ** 2), device=device).type(type_complex)

        for i, layer in enumerate(pmt):
            f_coeffs = fft_piecewise_constant(layer, fourier_order, device=device, type_complex=type_complex)
            center = torch.div(torch.tensor(f_coeffs.shape, device=device), 2, rounding_mode='trunc')

            conv_idx = torch.arange(-ff + 1, ff, 1, device=device).type(torch.long)
            conv_idx = circulant(conv_idx, device)
            conv_i = conv_idx.repeat_interleave(ff, dim=1).type(torch.long)
            conv_i = conv_i.repeat_interleave(ff, dim=0)
            conv_j = conv_idx.repeat(ff, ff).type(torch.long)
            e_conv = f_coeffs[center[0] + conv_i, center[1] + conv_j]
            res[i] = e_conv
    return res


def to_conv_mat_discrete(pmt, fourier_order, device=torch.device('cpu'), type_complex=torch.complex128, improve_dft=True):

    if len(pmt.shape) == 2:
        print('shape is 2')
        raise ValueError

    ff = 2 * fourier_order + 1

    if pmt.shape[1] == 1:  # 1D
        res = torch.zeros((pmt.shape[0], ff, ff), device=device).type(type_complex)
        if improve_dft:
            minimum_pattern_size = 2 * ff * pmt.shape[2]
        else:
            minimum_pattern_size = 2 * ff
        for i, layer in enumerate(pmt):
            n = minimum_pattern_size // layer.shape[1]
            layer = layer.repeat_interleave(n + 1, axis=1)

            f_coeffs = torch.fft.fftshift(torch.fft.fftn(layer / (layer.size(0)*layer.size(1))))
            center = f_coeffs.shape[1] // 2

            conv_idx = torch.arange(-ff + 1, ff, 1, device=device).type(torch.long)
            conv_idx = circulant(conv_idx, device)
            e_conv = f_coeffs[0, center + conv_idx]
            res[i] = e_conv

    else:  # 2D
        res = torch.zeros((pmt.shape[0], ff ** 2, ff ** 2), device=device).type(type_complex)
        if improve_dft:
            minimum_pattern_size_1 = 2 * ff * pmt.shape[1]
            minimum_pattern_size_2 = 2 * ff * pmt.shape[2]
        else:
            minimum_pattern_size_1 = 2 * ff
            minimum_pattern_size_2 = 2 * ff
        # 9 * (40*500) * (40*500) / 1E6 = 3600 MB = 3.6 GB
        for i, layer in enumerate(pmt):
            if layer.shape[0] < minimum_pattern_size_1:
                n = torch.div(minimum_pattern_size_1, layer.shape[0], rounding_mode='trunc')
                layer = layer.repeat_interleave(n + 1, axis=0)
            if layer.shape[1] < minimum_pattern_size_2:
                n = torch.div(minimum_pattern_size_2, layer.shape[1], rounding_mode='trunc')
                layer = layer.repeat_interleave(n + 1, axis=1)

            f_coeffs = torch.fft.fftshift(torch.fft.fft2(layer / (layer.size(0)*layer.size(1))))
            center = torch.div(torch.tensor(f_coeffs.shape, device=device), 2, rounding_mode='trunc')

            conv_idx = torch.arange(-ff + 1, ff, 1, device=device).type(torch.long)
            conv_idx = circulant(conv_idx, device)

            conv_i = conv_idx.repeat_interleave(ff, dim=1).type(torch.long)
            conv_i = conv_i.repeat_interleave(ff, dim=0)
            conv_j = conv_idx.repeat(ff, ff).type(torch.long)
            e_conv = f_coeffs[center[0] + conv_i, center[1] + conv_j]
            res[i] = e_conv
    return res


def circulant(c, device=torch.device('cpu')):

    center = c.shape[0] // 2
    circ = torch.zeros((center + 1, center + 1), device=device).type(torch.long)

    for r in range(center+1):
        idx = torch.arange(r, r - center - 1, -1, device=device)

        assign_value = c[center + idx]
        circ[r] = assign_value

    return circ
