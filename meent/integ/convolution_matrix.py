import meent.integ.backend.meentpy as ee

from os import walk
from scipy.io import loadmat
from pathlib import Path


def put_permittivity_in_ucell(ucell, mat_list, mat_table, wl):

    res = ee.zeros(ucell.shape, dtype='complex')

    for z in range(ucell.shape[0]):
        for y in range(ucell.shape[1]):
            for x in range(ucell.shape[2]):
                material = mat_list[ucell[z, y, x]]
                if type(material) == str:
                    # res[z, y, x] = find_nk_index(material, mat_table, wavelength) ** 2
                    assign_index = [z, y, x]
                    assign_value = find_nk_index(material, mat_table, wl) ** 2
                    res = ee.assign(res, assign_index, assign_value)

                else:
                    # res[z, y, x] = material ** 2
                    assign_index = [z, y, x]
                    assign_value = material ** 2

                    res = ee.assign(res, assign_index, assign_value)

    return res


def put_permittivity_in_ucell_object(ucell_size, mat_list, obj_list, mat_table, wl):
    # TODO: under development
    res = ee.zeros(ucell_size, dtype='complex')

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

    n_index = ee.interp(wl, mat_data[:, 0], mat_data[:, 1])

    if n_only:
        return n_index

    k_index = ee.interp(wl, mat_data[:, 0], mat_data[:, 2])
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
            data = ee.loadtxt(path, skiprows=1)
            mat_table[name[:-4].upper()] = data

        elif name[-3:] == 'mat':
            data = loadmat(path)
            data = ee.array([data['WL'], data['n'], data['k']])[:, :, 0].T
            mat_table[name[:-4].upper()] = data
    return mat_table


def cell_compression(cell):
    # find discontinuities in x
    step_y, step_x = 1. / ee.array(cell.shape)
    x = []
    y = []
    cell_x = []
    cell_xy = []

    cell_next = ee.roll(cell, -1, axis=1)

    for col in range(cell.shape[1]):
        if not (cell[:, col] == cell_next[:, col]).all() or (col == cell.shape[1] - 1):
            x.append(step_x * (col + 1))
            cell_x.append(cell[:, col])

    cell_x = ee.array(cell_x).T
    cell_x_next = ee.roll(cell_x, -1, axis=0)

    for row in range(cell_x.shape[0]):
        if not (cell_x[row, :] == cell_x_next[row, :]).all() or (row == cell_x.shape[0] - 1):
            y.append(step_y * (row + 1))
            cell_xy.append(cell_x[row, :])

    x = ee.array(x).reshape((-1, 1))
    y = ee.array(y).reshape((-1, 1))
    cell_comp = ee.array(cell_xy)

    return cell_comp, x, y


def fft_piecewise_constant(cell, fourier_order):
    if cell.shape[0] == 1:
        fourier_order = [0, fourier_order]
    else:
        fourier_order = [fourier_order, fourier_order]
    cell, x, y = cell_compression(cell)

    # X axis
    cell_next_x = ee.roll(cell, -1, axis=1)
    cell_diff_x = cell_next_x - cell

    modes = ee.arange(-2 * fourier_order[1], 2 * fourier_order[1] + 1, 1)

    f_coeffs_x = cell_diff_x @ ee.exp(-1j * 2 * ee.pi * x @ modes[None, :])
    c = f_coeffs_x.shape[1] // 2

    x_next = ee.vstack((ee.roll(x, -1, axis=0)[:-1], 1)) - x

    # f_coeffs_x[:, c] = (cell @ ee.vstack((x[0], x_next[:-1]))).flatten()

    assign_index = [ee.arange(len(f_coeffs_x)), ee.array([c])]
    assign_value = (cell @ ee.vstack((x[0], x_next[:-1]))).flatten()

    f_coeffs_x = ee.assign(f_coeffs_x, assign_index, assign_value)

    mask = ee.ones(f_coeffs_x.shape[1], dtype=bool)
    # mask[c] = False

    mask = ee.assign(mask, c, False)


    # f_coeffs_x[:, mask] /= (1j * 2 * ee.pi * modes[mask])

    assign_index = mask
    assign_value = f_coeffs_x[:, mask] / (1j * 2 * ee.pi * modes[mask])

    f_coeffs_x = ee.assign(f_coeffs_x, assign_index, assign_value, row_all=True)

    # Y axis
    f_coeffs_x_next_y = ee.roll(f_coeffs_x, -1, axis=0)
    f_coeffs_x_diff_y = f_coeffs_x_next_y - f_coeffs_x

    modes = ee.arange(-2 * fourier_order[0], 2 * fourier_order[0] + 1, 1)

    f_coeffs_xy = f_coeffs_x_diff_y.T @ ee.exp(-1j * 2 * ee.pi * y @ modes[None, :])
    c = f_coeffs_xy.shape[1] // 2

    y_next = ee.vstack((ee.roll(y, -1, axis=0)[:-1], 1)) - y

    # f_coeffs_xy[:, c] = f_coeffs_x.T @ ee.vstack((y[0], y_next[:-1])).flatten()

    assign_value = f_coeffs_x.T @ ee.vstack((y[0], y_next[:-1])).flatten()
    f_coeffs_xy = ee.assign(f_coeffs_xy, c, assign_value, row_all=True)

    if c:
        mask = ee.ones(f_coeffs_xy.shape[1], dtype=bool)
        # mask[c] = False
        mask = ee.assign(mask, c, False)

        # f_coeffs_xy[:, mask] /= (1j * 2 * ee.pi * modes[mask])

        assign_value = f_coeffs_xy[:, mask] / (1j * 2 * ee.pi * modes[mask])
        f_coeffs_xy = ee.assign(f_coeffs_xy, mask, assign_value, row_all=True)

    return f_coeffs_xy.T


def to_conv_mat(pmt, fourier_order):

    if len(pmt.shape) == 2:
        print('shape is 2')
        raise ValueError
    ff = 2 * fourier_order + 1

    if pmt.shape[1] == 1:  # 1D

        res = ee.zeros((pmt.shape[0], ff, ff)).astype('complex')

        for i, layer in enumerate(pmt):
            # f_coeffs = fft_piecewise_constant(layer, fourier_order)
            # A = ee.roll(circulant(f_coeffs.flatten()), (f_coeffs.size + 1) // 2, 0)
            # res[i] = A[:2 * fourier_order + 1, :2 * fourier_order + 1]
            f_coeffs = fft_piecewise_constant(layer, fourier_order)

            center = f_coeffs.shape[1] // 2

            conv_idx = ee.arange(-ff + 1, ff, 1)
            conv_idx = circulant(conv_idx)

            e_conv = f_coeffs[0, center + conv_idx]
            # res = res.at[i].set(e_conv)
            res = ee.assign(res, i, e_conv)

    else:  # 2D
        # attention on the order of axis (Z Y X)

        # TODO: separate fourier order
        res = ee.zeros((pmt.shape[0], ff ** 2, ff ** 2)).astype('complex')

        for i, layer in enumerate(pmt):
            # pmtvy_fft = fft_piecewise_constant(layer, fourier_order)
            #
            # center = ee.array(pmtvy_fft.shape) // 2
            #
            # conv_idx = ee.arange(-ff + 1, ff, 1)
            # conv_idx = circulant(conv_idx)[ff - 1:, :ff]
            #
            # conv_i = ee.repeat(conv_idx, ff, axis=1)
            # conv_i = ee.repeat(conv_i, [ff] * ff, axis=0)
            # conv_j = ee.tile(conv_idx, (ff, ff))
            # res[i] = pmtvy_fft[center[0] + conv_i, center[1] + conv_j]
            f_coeffs = fft_piecewise_constant(layer, fourier_order)

            center = ee.array(f_coeffs.shape) // 2

            conv_idx = ee.arange(-ff + 1, ff, 1)

            conv_idx = circulant(conv_idx)

            conv_i = ee.repeat(conv_idx, ff, axis=1)
            conv_i = ee.repeat(conv_i, ff, axis=0)
            conv_j = ee.tile(conv_idx, (ff, ff))

            # res = res.at[i].set(f_coeffs[center[0] + conv_i, center[1] + conv_j])
            assign_value = f_coeffs[center[0] + conv_i, center[1] + conv_j]
            res = ee.assign(res, i, assign_value)
    # import matplotlib.pyplot as plt
    #
    # plt.figure()
    # plt.imshow(abs(res[0]), cmap='jet')
    # plt.colorbar()
    # plt.show()
    #
    return res


# def draw_fill_factor(patterns_fill_factor, grating_type, resolution=1000, mode=0):
#
#     # res in Z X Y
#     if grating_type == 2:
#         res = ee.zeros((len(patterns_fill_factor), resolution, resolution), dtype='complex')
#     else:
#         res = ee.zeros((len(patterns_fill_factor), 1, resolution), dtype='complex')
#
#     if grating_type in (0, 1):  # TODO: handle this by len(fill_factor)
#         # fill_factor is not exactly implemented.
#         for i, (n_ridge, n_groove, fill_factor) in enumerate(patterns_fill_factor):
#             permittivity = ee.ones((1, resolution), dtype='complex')
#             cut = int(resolution * fill_factor)
#             permittivity[0, :cut] *= n_ridge ** 2
#             permittivity[0, cut:] *= n_groove ** 2
#             res[i, 0] = permittivity
#     else:  # 2D
#         for i, (n_ridge, n_groove, fill_factor) in enumerate(patterns_fill_factor):
#             fill_factor = ee.array(fill_factor)
#             permittivity = ee.ones((resolution, resolution), dtype='complex')
#             cut = (resolution * fill_factor)  # TODO: need parenthesis?
#             permittivity *= n_groove ** 2
#             permittivity[:int(cut[1]), :int(cut[0])] *= n_ridge ** 2
#             res[i] = permittivity
#
#     return res

def circulant(c):

    center = ee.array(c.shape) // 2
    circ = ee.zeros((center[0] + 1, center[0] + 1), dtype='int32')

    for r in range(center[0]+1):
        idx = ee.arange(r, r - center - 1, -1)

        # circ = circ.at[r].set(c[center + idx])
        assign_value = c[center + idx]
        circ = ee.assign(circ, r, assign_value)

    return circ
