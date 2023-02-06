import torch
import numpy as np

from os import walk
from scipy.io import loadmat
from pathlib import Path


def put_permittivity_in_ucell(ucell, mat_list, mat_table, wl, device=torch.device('cpu'), type_complex=torch.complex128):

    res = torch.zeros(ucell.shape, device=device).type(type_complex)

    for z in range(ucell.shape[0]):
        for y in range(ucell.shape[1]):
            for x in range(ucell.shape[2]):
                material = mat_list[ucell[z, y, x]]
                if type(material) == str:
                    res[z, y, x] = find_nk_index(material, mat_table, wl) ** 2
                else:
                    res[z, y, x] = material ** 2

    return res


def put_permittivity_in_ucell_object(ucell_size, mat_list, obj_list, mat_table, wl, device=torch.device('cpu'),
                                     type_complex=torch.complex128):
    # TODO: under development
    res = torch.zeros(ucell_size, device=device).type(type_complex)

    for material, obj_index in zip(mat_list, obj_list):
        if type(material) == str:
            res[obj_index] = find_nk_index(material, mat_table, wl) ** 2
        else:
            res[obj_index] = material ** 2

    return res


def find_nk_index(material, mat_table, wl):
    if material[-6:] == '__real':
        material = material[:-6]
        n_only = True
    else:
        n_only = False

    mat_data = mat_table[material.upper()]

    n_index = np.interp(wl, mat_data[:, 0], mat_data[:, 1])

    if n_only:
        return n_index

    k_index = np.interp(wl, mat_data[:, 0], mat_data[:, 2])
    nk = n_index + 1j * k_index

    return nk


def read_material_table(nk_path=None):
    mat_table = {}

    if nk_path is None:
        nk_path = str(Path(__file__).resolve().parent.parent) + '/nk_data'

    full_path_list, name_list, _ = [], [], []
    for (dirpath, dirnames, filenames) in walk(nk_path):
        full_path_list.extend([f'{dirpath}/{filename}' for filename in filenames])
        name_list.extend(filenames)
    for path, name in zip(full_path_list, name_list):
        if name[-3:] == 'txt':
            data = np.loadtxt(path, skiprows=1)
            mat_table[name[:-4].upper()] = data

        elif name[-3:] == 'mat':
            data = loadmat(path)
            data = np.array([data['WL'], data['n'], data['k']])[:, :, 0].T
            mat_table[name[:-4].upper()] = data
    return mat_table


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
    # cell_xa = torch.cat(cell_x, dim=0)
    # cell_xaa = torch.cat(cell_x, dim=1)
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


def to_conv_mat_piecewise_constant(pmt, fourier_order, device=torch.device('cpu'), type_complex=torch.complex128):

    if len(pmt.shape) == 2:
        print('shape is 2')
        raise ValueError
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

    # import matplotlib.pyplot as plt
    #
    # plt.figure()
    # plt.imshow(abs(res[0]), cmap='jet')
    # plt.colorbar()
    # plt.show()
    #
    return res


def to_conv_mat(pmt, fourier_order, device=torch.device('cpu'), type_complex=torch.complex128):

    if len(pmt.shape) == 2:
        print('shape is 2')
        raise ValueError
    ff = 2 * fourier_order + 1

    if pmt.shape[1] == 1:  # 1D
        res = torch.zeros((pmt.shape[0], ff, ff), device=device).type(type_complex)

        # extend array for FFT
        minimum_pattern_size = 2 * ff
        if pmt.shape[2] < minimum_pattern_size:
            n = minimum_pattern_size // pmt.shape[2]
            pmt = pmt.repeat_interleave(n+1, dim=2)

        for i, layer in enumerate(pmt):
            f_coeffs = torch.fft.fftshift(torch.fft.fftn(layer / (layer.size(0)*layer.size(1))))
            center = f_coeffs.shape[1] // 2

            conv_idx = torch.arange(-ff + 1, ff, 1, device=device).type(torch.long)
            conv_idx = circulant(conv_idx, device)
            e_conv = f_coeffs[0, center + conv_idx]
            res[i] = e_conv

    else:  # 2D
        res = torch.zeros((pmt.shape[0], ff ** 2, ff ** 2), device=device).type(type_complex)

        # extend array
        minimum_pattern_size = 2 * ff
        if pmt.shape[1] < minimum_pattern_size:
            n = minimum_pattern_size // pmt.shape[1]
            pmt = pmt.repeat_interleave(n+1, dim=1)
        if pmt.shape[2] < minimum_pattern_size:
            n = minimum_pattern_size // pmt.shape[2]
            pmt = pmt.repeat_interleave(n+1, dim=2)

        for i, layer in enumerate(pmt):
            f_coeffs = torch.fft.fftshift(torch.fft.fft2(layer / (layer.size(0)*layer.size(1))))
            center = torch.div(torch.tensor(f_coeffs.shape, device=device), 2, rounding_mode='trunc')

            conv_idx = torch.arange(-ff + 1, ff, 1, device=device).type(torch.long)
            conv_idx = circulant(conv_idx, device)

            conv_i = conv_idx.repeat_interleave(ff, dim=1).type(torch.long)
            conv_i = conv_i.repeat_interleave(ff, dim=0)
            conv_j = conv_idx.repeat(ff, ff).type(torch.long)
            e_conv = f_coeffs[center[0] + conv_i, center[1] + conv_j]
            res[i] = e_conv

    # import matplotlib.pyplot as plt
    #
    # plt.figure()
    # plt.imshow(abs(res[0]), cmap='jet')
    # plt.colorbar()
    # plt.show()

    return res


def circulant(c, device=torch.device('cpu')):
    # TODO: need device?

    center = c.shape[0] // 2
    circ = torch.zeros((center + 1, center + 1), device=device).type(torch.long)

    for r in range(center+1):
        idx = torch.arange(r, r - center - 1, -1, device=device)

        assign_value = c[center + idx]
        circ[r] = assign_value

    return circ
