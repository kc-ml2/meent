import copy
import jax.numpy as jnp

from os import walk
from scipy.io import loadmat
from pathlib import Path
# from jax.scipy.linalg import circulant  # hope this is supported

# TODO: whole code
# def put_n_ridge_in_pattern_fill_factor(pattern_all, mat_table, wavelength):
#
#     pattern_all = copy.deepcopy(pattern_all)
#
#     for i, (n_ridge, n_groove, pattern) in enumerate(pattern_all):
#
#         if type(n_ridge) == str:
#             material = n_ridge
#             n_ridge = find_nk_index(material, mat_table, wavelength)
#         pattern_all[i][0] = n_ridge
#     return pattern_all


# def get_material_index_in_ucell(ucell_comp, mat_list):
#
#     res = [[[] for _ in mat_list] for _ in ucell_comp]
#
#     for z, ucell_xy in enumerate(ucell_comp):
#         for y in range(ucell_xy.shape[0]):
#             for x in range(ucell_xy.shape[1]):
#                 res[z][ucell_xy[y, x]].append([y, x])
#     return res


# def put_permittivity_in_ucell_object_comps(ucell, mat_list, obj_list, mat_table, wavelength):
#
#     res = np.zeros(ucell.shape, dtype='complex')
#
#     for obj_xy in obj_list:
#         for material, obj_index in zip(mat_list, obj_xy):
#             obj_index = np.array(obj_index).T
#             if type(material) == str:
#                 res[obj_index[0], obj_index[1]] = find_nk_index(material, mat_table, wavelength) ** 2
#             else:
#                 res[obj_index[0], obj_index[1]] = material ** 2
#
#     return res


def put_permittivity_in_ucell(ucell, mat_list, mat_table, wl):

    res = jnp.zeros(ucell.shape, dtype='complex')

    for z in range(ucell.shape[0]):
        for y in range(ucell.shape[1]):
            for x in range(ucell.shape[2]):
                material = mat_list[ucell[z, y, x]]
                if type(material) == str:
                    # res[z, y, x] = find_nk_index(material, mat_table, wavelength) ** 2
                    res = res.at[z, y, x].set(find_nk_index(material, mat_table, wl) ** 2)
                else:
                    # res[z, y, x] = material ** 2
                    res = res.at[z, y, x].set(material ** 2)

    return res


def put_permittivity_in_ucell_object(ucell_size, mat_list, obj_list, mat_table, wl):
    # TODO: under development
    res = jnp.zeros(ucell_size, dtype='complex')

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

    n_index = jnp.interp(wl, mat_data[:, 0], mat_data[:, 1])

    if n_only:
        return n_index

    k_index = jnp.interp(wl, mat_data[:, 0], mat_data[:, 2])
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
            import numpy
            data = numpy.loadtxt(path, skiprows=1)
            mat_table[name[:-4].upper()] = data

        elif name[-3:] == 'mat':
            data = loadmat(path)
            data = jnp.array([data['WL'], data['n'], data['k']])[:, :, 0].T
            mat_table[name[:-4].upper()] = data
    return mat_table


def cell_compression(cell):
    # find discontinuities in x
    step_y, step_x = 1. / jnp.array(cell.shape)
    x = []
    y = []
    cell_x = []
    cell_xy = []

    cell_next = jnp.roll(cell, -1, axis=1)

    for col in range(cell.shape[1]):
        if not (cell[:, col] == cell_next[:, col]).all() or (col == cell.shape[1] - 1):
            x.append(step_x * (col + 1))
            cell_x.append(cell[:, col])

    cell_x = jnp.array(cell_x).T
    cell_x_next = jnp.roll(cell_x, -1, axis=0)

    for row in range(cell_x.shape[0]):
        if not (cell_x[row, :] == cell_x_next[row, :]).all() or (row == cell_x.shape[0] - 1):
            y.append(step_y * (row + 1))
            cell_xy.append(cell_x[row, :])

    x = jnp.array(x).reshape((-1, 1))
    y = jnp.array(y).reshape((-1, 1))
    cell_comp = jnp.array(cell_xy)

    return cell_comp, x, y


def fft_piecewise_constant(cell, fourier_order):
    if cell.shape[0] == 1:
        fourier_order = [0, fourier_order]
    else:
        fourier_order = [fourier_order, fourier_order]
    cell, x, y = cell_compression(cell)

    # X axis
    cell_next_x = jnp.roll(cell, -1, axis=1)
    cell_diff_x = cell_next_x - cell

    modes = jnp.arange(-2 * fourier_order[1], 2 * fourier_order[1] + 1, 1)

    f_coeffs_x = cell_diff_x @ jnp.exp(-1j * 2 * jnp.pi * x @ modes[None, :])
    c = f_coeffs_x.shape[1] // 2

    x_next = jnp.vstack((jnp.roll(x, -1, axis=0)[:-1], 1)) - x

    try:
        f_coeffs_x[:, c] = (cell @ jnp.vstack((x[0], x_next[:-1]))).flatten()
    except:
        row, _ = f_coeffs_x.shape
        f_coeffs_x = f_coeffs_x.at[:, c].set((cell @ jnp.vstack((x[0], x_next[:-1]))).flatten())
    if c:
        mask = jnp.ones(f_coeffs_x.shape[1], dtype=bool)

        try:
            mask[c] = False
            f_coeffs_x[:, mask] /= (1j * 2 * jnp.pi * modes[mask])

        except:
            mask = mask.at[c].set(False)
            temp = f_coeffs_x[:, mask] / (1j * 2 * jnp.pi * modes[mask])
            f_coeffs_x = f_coeffs_x.at[:, mask].set(temp)

    # Y axis
    f_coeffs_x_next_y = jnp.roll(f_coeffs_x, -1, axis=0)
    f_coeffs_x_diff_y = f_coeffs_x_next_y - f_coeffs_x

    modes = jnp.arange(-2 * fourier_order[0], 2 * fourier_order[0] + 1, 1)

    f_coeffs_xy = f_coeffs_x_diff_y.T @ jnp.exp(-1j * 2 * jnp.pi * y @ modes[None, :])
    c = f_coeffs_xy.shape[1] // 2

    y_next = jnp.vstack((jnp.roll(y, -1, axis=0)[:-1], 1)) - y
    try:
        f_coeffs_xy[:, c] = f_coeffs_x.T @ jnp.vstack((y[0], y_next[:-1])).flatten()
    except:
        # row, _ = f_coeffs_xy.shape
        f_coeffs_xy = f_coeffs_xy.at[:, c].set(f_coeffs_x.T @ jnp.vstack((y[0], y_next[:-1])).flatten())

    if c:
        mask = jnp.ones(f_coeffs_xy.shape[1], dtype=bool)

        try:
            mask[c] = False
            f_coeffs_xy[:, mask] /= (1j * 2 * jnp.pi * modes[mask])

        except:
            mask = mask.at[c].set(False)
            temp = f_coeffs_xy[:, mask] / (1j * 2 * jnp.pi * modes[mask])
            f_coeffs_xy = f_coeffs_xy.at[:, mask].set(temp)

    return f_coeffs_xy.T


def to_conv_mat(pmt, fourier_order):

    if len(pmt.shape) == 2:
        print('shape is 2')
        raise ValueError
    ff = 2 * fourier_order + 1

    if pmt.shape[1] == 1:  # 1D

        res = jnp.zeros((pmt.shape[0], ff, ff)).astype('complex')

        for i, layer in enumerate(pmt):
            f_coeffs = fft_piecewise_constant(layer, fourier_order)

            center = f_coeffs.shape[1] // 2

            conv_idx = jnp.arange(ff - 1, -ff, -1)
            conv_idx = circulant(conv_idx)

            e_conv = f_coeffs[0, center + conv_idx]
            res = res.at[i].set(e_conv)

    else:  # 2D
        # attention on the order of axis (Z Y X)

        # TODO: separate fourier order
        res = jnp.zeros((pmt.shape[0], ff ** 2, ff ** 2)).astype('complex')

        for i, layer in enumerate(pmt):
            f_coeffs = fft_piecewise_constant(layer, fourier_order)

            center = jnp.array(f_coeffs.shape) // 2

            conv_idx = jnp.arange(-ff + 1, ff, 1)

            conv_idx = circulant(conv_idx)

            conv_i = jnp.repeat(conv_idx, ff, axis=1)
            conv_i = jnp.repeat(conv_i, ff, axis=0)
            conv_j = jnp.tile(conv_idx, (ff, ff))

            res = res.at[i].set(f_coeffs[center[0] + conv_i, center[1] + conv_j])

    # import matplotlib.pyplot as plt
    #
    # plt.figure()
    # plt.imshow(abs(res[0]), cmap='jet')
    # plt.colorbar()
    # plt.show()
    #
    return res



# def to_conv_mat_jax(pmt, fourier_order):
#     # FFT scaling: https://kr.mathworks.com/matlabcentral/answers/15770-scaling-the-fft-and-the-ifft?s_tid=srchtitle
#     if len(pmt.shape) == 2:
#         print('shape is 2')
#         raise ValueError
#     ff = 2 * fourier_order + 1
#
#     if pmt.shape[1] == 1:  # 1D
#
#         res = jnp.zeros((pmt.shape[0], ff, ff)).astype('complex')
#
#         # extend array for FFT
#         minimum_pattern_size = (4 * fourier_order + 1) * pmt.shape[2]
#         # TODO: what is theoretical minimum?
#         # TODO: can be a scalability issue
#         if pmt.shape[2] < minimum_pattern_size:
#             n = minimum_pattern_size // pmt.shape[2]
#             pmt = jnp.repeat(pmt, n + 1, axis=2)
#
#         for i, pmtvy in enumerate(pmt):
#             pmtvy_fft = jnp.fft.fftshift(jnp.fft.fftn(pmtvy / pmtvy.size))
#             center = pmtvy_fft.shape[1] // 2
#
#             conv_idx = jnp.arange(ff - 1, -ff, -1)
#             conv_idx = circulant(conv_idx)
#             res = res.at[i].set(pmtvy_fft[1, center + conv_idx])
#
#     else:  # 2D
#         # attention on the order of axis.
#         # Here X Y Z. Cell Drawing in CAD is Y X Z. Here is Z Y X
#
#         # TODO: separate fourier order
#         res = jnp.zeros((pmt.shape[0], ff ** 2, ff ** 2)).astype('complex')
#
#         # extend array
#         # TODO: run test
#         minimum_pattern_size = ff ** 2
#         # TODO: what is theoretical minimum?
#         # TODO: can be a scalability issue
#
#         if pmt.shape[1] < minimum_pattern_size:
#             n = minimum_pattern_size // pmt.shape[1]
#             pmt = jnp.repeat(pmt, n + 1, axis=1)
#         if pmt.shape[2] < minimum_pattern_size:
#             n = minimum_pattern_size // pmt.shape[2]
#             pmt = jnp.repeat(pmt, n + 1, axis=2)
#
#         for i, layer in enumerate(pmt):
#             pmtvy_fft = jnp.fft.fftshift(jnp.fft.fft2(layer / layer.size))
#
#             center = jnp.array(pmtvy_fft.shape) // 2
#
#             conv_idx = jnp.arange(ff - 1, -ff, -1)
#             conv_idx = circulant(conv_idx)
#
#             conv_i = jnp.repeat(conv_idx, ff, axis=1)
#             conv_i = jnp.repeat(conv_i, ff, axis=0)
#             conv_j = jnp.tile(conv_idx, (ff, ff))
#
#             res = res.at[i].set(pmtvy_fft[center[0] + conv_i, center[1] + conv_j])

    # import matplotlib.pyplot as plt
    #
    # plt.figure()
    # plt.imshow(abs(res[0]), cmap='jet')
    # plt.colorbar()
    # plt.show()
    #
    return res
#
#
# def draw_fill_factor(patterns_fill_factor, grating_type, resolution=1000):
#
#     # res in Z X Y
#     if grating_type == 2:
#         res = jnp.zeros((len(patterns_fill_factor), resolution, resolution), dtype='complex')
#     else:
#         res = jnp.zeros((len(patterns_fill_factor), 1, resolution), dtype='complex')
#
#     if grating_type in (0, 1):  # TODO: handle this by len(fill_factor)
#         # fill_factor is not exactly implemented.
#         for i, (n_ridge, n_groove, fill_factor) in enumerate(patterns_fill_factor):
#             permittivity = jnp.ones((1, resolution), dtype='complex')
#             cut = int(resolution * fill_factor)
#
#             cut_idx = jnp.arange(cut)
#             permittivity *= n_groove ** 2
#
#             permittivity = permittivity.at[0, cut_idx].set(n_ridge ** 2)
#             res = res.at[i].set(permittivity)
#
#     else:  # 2D
#         for i, (n_ridge, n_groove, fill_factor) in enumerate(patterns_fill_factor):
#             fill_factor = jnp.array(fill_factor)
#             permittivity = jnp.ones((resolution, resolution), dtype='complex')
#             cut = (resolution * fill_factor)  # TODO: need parenthesis?
#             cut_idx_row = jnp.arange(int(cut[1]))
#             cut_idx_column = jnp.arange(int(cut[0]))
#
#             permittivity *= n_groove ** 2
#
#             rows, cols = jnp.meshgrid(cut_idx_row, cut_idx_column, indexing='ij')
#
#             permittivity = permittivity.at[rows, cols].set(n_ridge ** 2)
#             res = res.at[i].set(permittivity)
#
#     return res


def circulant(c):

    center = jnp.array(c.shape) // 2
    circ = jnp.zeros((center[0] + 1, center[0] + 1), dtype='int32')

    for r in range(center[0]+1):
        idx = jnp.arange(r, r - center - 1, -1)

        circ = circ.at[r].set(c[center + idx])

    return circ
