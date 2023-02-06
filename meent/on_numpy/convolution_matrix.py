import copy
import time

import numpy as np

from os import walk
from scipy.io import loadmat
from pathlib import Path


def put_permittivity_in_ucell(ucell, mat_list, mat_table, wl, type_complex=np.complex128):

    res = np.zeros(ucell.shape, dtype=type_complex)

    for z in range(ucell.shape[0]):
        for y in range(ucell.shape[1]):
            for x in range(ucell.shape[2]):
                material = mat_list[ucell[z, y, x]]
                if type(material) == str:
                    res[z, y, x] = find_nk_index(material, mat_table, wl) ** 2
                else:
                    res[z, y, x] = material ** 2

    return res


def put_permittivity_in_ucell_object(ucell_size, mat_list, obj_list, mat_table, wl,
                                     type_complex=np.complex128):
    # TODO: under development
    res = np.zeros(ucell_size, dtype=type_complex)

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


def fft_piecewise_constant(cell, fourier_order, type_complex=np.complex128):
    """
    reference: reticolo
    """

    if cell.shape[0] == 1:
        fourier_order = [0, fourier_order]
    else:
        fourier_order = [fourier_order, fourier_order]
    cell, x, y = cell_compression(cell, type_complex=type_complex)

    # X axis
    cell_next_x = np.roll(cell, -1, axis=1)
    cell_diff_x = cell_next_x - cell

    modes = np.arange(-2 * fourier_order[1], 2 * fourier_order[1] + 1, 1)

    f_coeffs_x = cell_diff_x @ np.exp(-1j * 2 * np.pi * x @ modes[None, :], dtype=type_complex)
    c = f_coeffs_x.shape[1] // 2

    # x_next = np.vstack(np.roll(x, -1, axis=0)[:-1]) - x
    x_next = np.vstack((np.roll(x, -1, axis=0)[:-1], 1)) - x

    f_coeffs_x[:, c] = (cell @ np.vstack((x[0], x_next[:-1]))).flatten()
    mask = np.ones(f_coeffs_x.shape[1], dtype=bool)
    mask[c] = False
    f_coeffs_x[:, mask] /= (1j * 2 * np.pi * modes[mask])

    # Y axis
    f_coeffs_x_next_y = np.roll(f_coeffs_x, -1, axis=0)
    f_coeffs_x_diff_y = f_coeffs_x_next_y - f_coeffs_x

    modes = np.arange(-2 * fourier_order[0], 2 * fourier_order[0] + 1, 1)

    f_coeffs_xy = f_coeffs_x_diff_y.T @ np.exp(-1j * 2 * np.pi * y @ modes[None, :], dtype=type_complex)
    c = f_coeffs_xy.shape[1] // 2

    y_next = np.vstack((np.roll(y, -1, axis=0)[:-1], 1)) - y

    f_coeffs_xy[:, c] = f_coeffs_x.T @ np.vstack((y[0], y_next[:-1])).flatten()

    if c:
        mask = np.ones(f_coeffs_xy.shape[1], dtype=bool)
        mask[c] = False
        f_coeffs_xy[:, mask] /= (1j * 2 * np.pi * modes[mask])

    return f_coeffs_xy.T


def to_conv_mat_piecewise_constant(pmt, fourier_order, type_complex=np.complex128):
    # TODO: do conv and 1/conv in simultaneously?
    if len(pmt.shape) == 2:
        print('shape is 2')
        raise ValueError
    ff = 2 * fourier_order + 1

    if pmt.shape[1] == 1:  # 1D
        res = np.zeros((pmt.shape[0], ff, ff)).astype(type_complex)

        for i, layer in enumerate(pmt):
            f_coeffs = fft_piecewise_constant(layer, fourier_order, type_complex=type_complex)
            center = f_coeffs.shape[1] // 2
            conv_idx = np.arange(-ff + 1, ff, 1, dtype=int)
            conv_idx = circulant(conv_idx)
            e_conv = f_coeffs[0, center + conv_idx]
            res[i] = e_conv

    else:  # 2D
        # attention on the order of axis (Z Y X)
        res = np.zeros((pmt.shape[0], ff ** 2, ff ** 2)).astype(type_complex)

        for i, layer in enumerate(pmt):
            f_coeffs = fft_piecewise_constant(layer, fourier_order, type_complex=type_complex)
            center = np.array(f_coeffs.shape) // 2

            conv_idx = np.arange(-ff + 1, ff, 1)
            conv_idx = circulant(conv_idx)
            conv_i = np.repeat(conv_idx, ff, axis=1)
            conv_i = np.repeat(conv_i, [ff] * ff, axis=0)
            conv_j = np.tile(conv_idx, (ff, ff))
            e_conv = f_coeffs[center[0] + conv_i, center[1] + conv_j]
            res[i] = e_conv

    # import matplotlib.pyplot as plt
    #
    # plt.figure()
    # plt.imshow(abs(res[0]), cmap='jet')
    # plt.colorbar()
    # plt.show()

    # import matplotlib.pyplot as plt
    #
    # plt.figure()
    # plt.imshow(abs(pmtvy_fft), cmap='jet')
    # plt.colorbar()
    # plt.show()
    #
    # plt.figure()
    # plt.imshow(abs(res[0]), cmap='jet')
    # plt.colorbar()
    # plt.show()

    return res


def to_conv_mat(pmt, fourier_order, type_complex=np.complex128):
    # TODO: large minimum size shows good convergence to piecewise_constant method. Add an option?

    if len(pmt.shape) == 2:
        print('shape is 2')
        raise ValueError
    ff = 2 * fourier_order + 1

    if pmt.shape[1] == 1:  # 1D
        res = np.zeros((pmt.shape[0], ff, ff)).astype(type_complex)

        # extend array
        minimum_pattern_size = 2 * ff
        if pmt.shape[2] < minimum_pattern_size:
            n = minimum_pattern_size // pmt.shape[2]
            pmt = np.repeat(pmt, n+1, axis=2)

        for i, layer in enumerate(pmt):
            f_coeffs = np.fft.fftshift(np.fft.fftn(layer / layer.size))
            # FFT scaling:
            # https://kr.mathworks.com/matlabcentral/answers/15770-scaling-the-fft-and-the-ifft?s_tid=srchtitle

            center = f_coeffs.shape[1] // 2

            conv_idx = np.arange(-ff + 1, ff, 1, dtype=int)
            conv_idx = circulant(conv_idx)
            e_conv = f_coeffs[0, center + conv_idx]
            res[i] = e_conv

    else:  # 2D
        # attention on the order of axis (Z Y X)
        # TODO: separate fourier order
        res = np.zeros((pmt.shape[0], ff ** 2, ff ** 2)).astype(type_complex)

        # extend array
        minimum_pattern_size = 2 * ff

        if pmt.shape[1] < minimum_pattern_size:
            n = minimum_pattern_size // pmt.shape[1]
            pmt = np.repeat(pmt, n+1, axis=1)
        if pmt.shape[2] < minimum_pattern_size:
            n = minimum_pattern_size // pmt.shape[2]
            pmt = np.repeat(pmt, n+1, axis=2)

        for i, layer in enumerate(pmt):
            f_coeffs = np.fft.fftshift(np.fft.fft2(layer / layer.size))
            center = np.array(f_coeffs.shape) // 2

            conv_idx = np.arange(-ff + 1, ff, 1)
            conv_idx = circulant(conv_idx)

            conv_i = np.repeat(conv_idx, ff, axis=1)
            conv_i = np.repeat(conv_i, [ff] * ff, axis=0)
            conv_j = np.tile(conv_idx, (ff, ff))
            res[i] = f_coeffs[center[0] + conv_i, center[1] + conv_j]

    # import matplotlib.pyplot as plt
    #
    # plt.figure()
    # plt.imshow(abs(pmtvy_fft), cmap='jet')
    # plt.colorbar()
    # plt.show()
    #
    # plt.figure()
    # plt.imshow(abs(res[0]), cmap='jet')
    # plt.colorbar()
    # plt.show()

    #
    return res


def circulant(c):

    center = c.shape[0] // 2
    circ = np.zeros((center + 1, center + 1), dtype=int)

    for r in range(center+1):
        idx = np.arange(r, r - center - 1, -1, dtype=int)

        assign_value = c[center + idx]
        circ[r] = assign_value

    return circ
