import copy
import numpy as np

from os import walk
from scipy.io import loadmat
from scipy.linalg import circulant
from pathlib import Path


def put_n_ridge_in_pattern_fill_factor(pattern_all, mat_table, wl):

    pattern_all = copy.deepcopy(pattern_all)

    for i, (n_ridge, n_groove, pattern) in enumerate(pattern_all):

        if type(n_ridge) == str:
            material = n_ridge
            n_ridge = find_nk_index(material, mat_table, wl)
        pattern_all[i][0] = n_ridge
    return pattern_all


def get_material_index_in_ucell(ucell_comp, mat_list):

    res = [[[] for _ in mat_list] for _ in ucell_comp]

    for z, ucell_xy in enumerate(ucell_comp):
        for y in range(ucell_xy.shape[0]):
            for x in range(ucell_xy.shape[1]):
                res[z][ucell_xy[y, x]].append([y, x])
    return res


def put_permittivity_in_ucell_object_comps(ucell, mat_list, obj_list, mat_table, wl):

    res = np.zeros(ucell.shape, dtype='complex')

    for obj_xy in obj_list:
        for material, obj_index in zip(mat_list, obj_xy):
            obj_index = np.array(obj_index).T
            if type(material) == str:
                res[obj_index[0], obj_index[1]] = find_nk_index(material, mat_table, wl) ** 2
            else:
                res[obj_index[0], obj_index[1]] = material ** 2

    return res


def put_permittivity_in_ucell(ucell, mat_list, mat_table, wl):

    res = np.zeros(ucell.shape, dtype='complex')

    for z in range(ucell.shape[0]):
        for y in range(ucell.shape[1]):
            for x in range(ucell.shape[2]):
                material = mat_list[ucell[z, y, x]]
                if type(material) == str:
                    res[z, y, x] = find_nk_index(material, mat_table, wl) ** 2
                else:
                    res[z, y, x] = material ** 2

    return res


def put_permittivity_in_ucell_object(ucell_size, mat_list, obj_list, mat_table, wl):

    res = np.zeros(ucell_size, dtype='complex')

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


def fill_factor_to_ucell(patterns_fill_factor, wl, grating_type, mat_table):
    pattern_fill_factor = put_n_ridge_in_pattern_fill_factor(patterns_fill_factor, mat_table, wl)
    ucell = draw_fill_factor(pattern_fill_factor, grating_type)

    return ucell


def cell_compression(cell):
    # find discontinuities in x
    step_y, step_x = 1. / np.array(cell.shape)
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


def fft_piecewise_constant(cell, fourier_order):
    if cell.shape[0] == 1:
        fourier_order = [0, fourier_order]
    else:
        fourier_order = [fourier_order, fourier_order]
    cell, x, y = cell_compression(cell)

    # X axis
    cell_next_x = np.roll(cell, -1, axis=1)
    cell_diff_x = cell_next_x - cell

    modes = np.arange(-2 * fourier_order[1], 2 * fourier_order[1] + 1, 1)

    f_coeffs_x = cell_diff_x @ np.exp(-1j * 2 * np.pi * x @ modes[None, :])
    c = f_coeffs_x.shape[1] // 2

    x_next = np.vstack((np.roll(x, -1, axis=0)[:-1], 1)) - x

    f_coeffs_x[:, c] = (cell @ np.vstack((x[0], x_next[:-1]))).flatten()
    mask = np.ones(f_coeffs_x.shape[1], dtype=bool)
    mask[c] = False
    f_coeffs_x[:, mask] /= (1j * 2 * np.pi * modes[mask])

    # Y axis
    f_coeffs_x_next_y = np.roll(f_coeffs_x, -1, axis=0)
    f_coeffs_x_diff_y = f_coeffs_x_next_y - f_coeffs_x

    modes = np.arange(-2 * fourier_order[0], 2 * fourier_order[0] + 1, 1)

    f_coeffs_xy = f_coeffs_x_diff_y.T @ np.exp(-1j * 2 * np.pi * y @ modes[None, :])
    c = f_coeffs_xy.shape[1] // 2

    y_next = np.vstack((np.roll(y, -1, axis=0)[:-1], 1)) - y

    f_coeffs_xy[:, c] = f_coeffs_x.T @ np.vstack((y[0], y_next[:-1])).flatten()
    mask = np.ones(f_coeffs_xy.shape[1], dtype=bool)
    mask[c] = False
    f_coeffs_xy[:, mask] /= (1j * 2 * np.pi * modes[mask])

    return f_coeffs_xy.T


def to_conv_mat(pmt, fourier_order):

    if len(pmt.shape) == 2:
        print('shape is 2')
        raise ValueError
    ff = 2 * fourier_order + 1

    if pmt.shape[1] == 1:  # 1D

        res = np.zeros((pmt.shape[0], ff, ff)).astype('complex')

        for i, layer in enumerate(pmt):
            f_coeffs = fft_piecewise_constant(layer, fourier_order)
            A = np.roll(circulant(f_coeffs.flatten()), (f_coeffs.size + 1) // 2, 0)
            res[i] = A[:2 * fourier_order + 1, :2 * fourier_order + 1]

    else:  # 2D
        # attention on the order of axis (Z Y X)

        # TODO: separate fourier order
        res = np.zeros((pmt.shape[0], ff ** 2, ff ** 2)).astype('complex')

        for i, layer in enumerate(pmt):
            pmtvy_fft = fft_piecewise_constant(layer, fourier_order)

            center = np.array(pmtvy_fft.shape) // 2

            conv_idx = np.arange(ff - 1, -ff, -1)
            conv_idx = circulant(conv_idx)[ff - 1:, :ff]

            conv_i = np.repeat(conv_idx, ff, axis=1)
            conv_i = np.repeat(conv_i, [ff] * ff, axis=0)
            conv_j = np.tile(conv_idx, (ff, ff))
            res[i] = pmtvy_fft[center[0] + conv_i, center[1] + conv_j]

    # import matplotlib.pyplot as plt
    #
    # plt.figure()
    # plt.imshow(abs(res[0]), cmap='jet')
    # plt.colorbar()
    # plt.show()

    return res


def draw_fill_factor(patterns_fill_factor, grating_type, resolution=1000, mode=0):

    # res in Z X Y
    if grating_type == 2:
        res = np.zeros((len(patterns_fill_factor), resolution, resolution), dtype='complex')
    else:
        res = np.zeros((len(patterns_fill_factor), 1, resolution), dtype='complex')

    if grating_type in (0, 1):  # TODO: handle this by len(fill_factor)
        # fill_factor is not exactly implemented.
        for i, (n_ridge, n_groove, fill_factor) in enumerate(patterns_fill_factor):
            permittivity = np.ones((1, resolution), dtype='complex')
            cut = int(resolution * fill_factor)
            permittivity[0, :cut] *= n_ridge ** 2
            permittivity[0, cut:] *= n_groove ** 2
            res[i, 0] = permittivity
    else:  # 2D
        for i, (n_ridge, n_groove, fill_factor) in enumerate(patterns_fill_factor):
            fill_factor = np.array(fill_factor)
            permittivity = np.ones((resolution, resolution), dtype='complex')
            cut = (resolution * fill_factor)  # TODO: need parenthesis?
            permittivity *= n_groove ** 2
            permittivity[:int(cut[1]), :int(cut[0])] *= n_ridge ** 2
            res[i] = permittivity

    return res
