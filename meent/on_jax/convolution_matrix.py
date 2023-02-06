import time
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

# import meent.on_jax.jitted as ee
# import jitted as ee
from . import jitted as ee

from os import walk
from scipy.io import loadmat
from pathlib import Path


def put_permittivity_in_ucell(ucell, mat_list, mat_table, wl, type_complex=jnp.complex128):

    res = ee.zeros(ucell.shape, dtype=type_complex)

    for z in range(ucell.shape[0]):
        for y in range(ucell.shape[1]):
            for x in range(ucell.shape[2]):
                material = mat_list[int(ucell[z, y, x])]
                assign_index = (z, y, x)

                if type(material) == str:
                    assign_value = find_nk_index(material, mat_table, wl, type_complex=type_complex) ** 2
                else:
                    assign_value = type_complex(material ** 2)  # TODO: need type complex?

                # res = res.at[assign_index].set(assign_value)
                res = ee.assign(res, assign_index, assign_value)

    return res


def put_permittivity_in_ucell_object(ucell_size, mat_list, obj_list, mat_table, wl,
                                     type_complex=jnp.complex128):
    # TODO: under development
    res = ee.zeros(ucell_size, dtype=type_complex)

    for material, obj_index in zip(mat_list, obj_list):
        if type(material) == str:
            res[obj_index] = find_nk_index(material, mat_table, wl, type_complex=type_complex) ** 2
        else:
            res[obj_index] = material ** 2

    return res


def find_nk_index(material, mat_table, wl, type_complex=jnp.complex128):
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
    nk = (n_index + 1j * k_index).astype(type_complex)

    return nk


def read_material_table(nk_path=None, type_complex=jnp.complex128):

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
            mat_table[name[:-4].upper()] = type_complex(data)

        elif name[-3:] == 'mat':
            data = loadmat(path)

            # TODO: need astype?
            # data = ee.array([data['WL'], data['n'], data['k']])[:, :, 0].T
            data['WL'], data['n'], data['k'] = data['WL'].astype(type_complex), data['n'].astype(type_complex), data['k'].astype(type_complex)

            data = ee.array([data['WL'], data['n'], data['k']])[:, :, 0].T
            mat_table[name[:-4].upper()] = data
    return mat_table


# can't jit
def cell_compression(cell, type_complex=jnp.complex128):

    if type_complex == jnp.complex128:
        type_float = jnp.float64
    else:
        type_float = jnp.float32

    # find discontinuities in x
    step_y, step_x = 1. / ee.array(cell.shape, dtype=type_float)
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


# @partial(jax.jit, static_argnums=(1,2 ))
def fft_piecewise_constant(cell, fourier_order, type_complex=jnp.complex128):

    if cell.shape[0] == 1:
        fourier_order = [0, fourier_order]
    else:
        fourier_order = [fourier_order, fourier_order]
    cell, x, y = cell_compression(cell, type_complex=type_complex)

    # X axis
    cell_next_x = ee.roll(cell, -1, axis=1)
    cell_diff_x = cell_next_x - cell

    modes = ee.arange(-2 * fourier_order[1], 2 * fourier_order[1] + 1, 1)

    f_coeffs_x = cell_diff_x @ ee.exp(-1j * 2 * ee.pi * x @ modes[None, :]).astype(type_complex)
    c = f_coeffs_x.shape[1] // 2

    x_next = ee.vstack((ee.roll(x, -1, axis=0)[:-1], 1)) - x

    assign_index = (ee.arange(len(f_coeffs_x)), ee.array([c]))
    assign_value = (cell @ ee.vstack((x[0], x_next[:-1]))).flatten().astype(type_complex)

    f_coeffs_x = ee.assign(f_coeffs_x, assign_index, assign_value)
    # f_coeffs_x = f_coeffs_x.at[assign_index].set(assign_value)

    mask_int = ee.hstack([ee.arange(c), ee.arange(c+1, f_coeffs_x.shape[1])])

    assign_index = mask_int

    assign_value = f_coeffs_x[:, mask_int] / (1j * 2 * ee.pi * modes[mask_int])

    f_coeffs_x = ee.assign(f_coeffs_x, assign_index, assign_value, row_all=True)
    # f_coeffs_x = f_coeffs_x.at[:, assign_index].set(assign_value)

    # Y axis
    f_coeffs_x_next_y = ee.roll(f_coeffs_x, -1, axis=0)
    f_coeffs_x_diff_y = f_coeffs_x_next_y - f_coeffs_x

    modes = ee.arange(-2 * fourier_order[0], 2 * fourier_order[0] + 1, 1)

    f_coeffs_xy = f_coeffs_x_diff_y.T @ ee.exp(-1j * 2 * ee.pi * y @ modes[None, :]).astype(type_complex)
    c = f_coeffs_xy.shape[1] // 2

    y_next = ee.vstack((ee.roll(y, -1, axis=0)[:-1], 1)) - y

    assign_index = [c]
    assign_value = f_coeffs_x.T @ ee.vstack((y[0], y_next[:-1])).astype(type_complex)
    f_coeffs_xy = ee.assign(f_coeffs_xy, assign_index, assign_value, row_all=True)
    # f_coeffs_xy = f_coeffs_xy.at[:, assign_index].set(assign_value)


    if c:
        mask_int = ee.hstack([ee.arange(c), ee.arange(c + 1, f_coeffs_x.shape[1])])

        assign_index = mask_int
        assign_value = f_coeffs_xy[:, mask_int] / (1j * 2 * ee.pi * modes[mask_int])

        f_coeffs_xy = ee.assign(f_coeffs_xy, assign_index, assign_value, row_all=True)
        # f_coeffs_xy = f_coeffs_xy.at[:, assign_index].set(assign_value)

    return f_coeffs_xy.T


def to_conv_mat_piecewise_constant(pmt, fourier_order, type_complex=jnp.complex128):

    if len(pmt.shape) == 2:
        print('shape is 2')
        raise ValueError
    ff = 2 * fourier_order + 1

    if pmt.shape[1] == 1:  # 1D
        res = ee.zeros((pmt.shape[0], ff, ff)).astype(type_complex)

        for i, layer in enumerate(pmt):
            f_coeffs = fft_piecewise_constant(layer, fourier_order, type_complex=type_complex)
            center = f_coeffs.shape[1] // 2
            conv_idx = ee.arange(-ff + 1, ff, 1)
            conv_idx = circulant(conv_idx)
            e_conv = f_coeffs[0, center + conv_idx]
            res = res.at[i].set(e_conv)
            # res = ee.assign(res, i, e_conv)

    else:  # 2D
        # attention on the order of axis (Z Y X)
        res = ee.zeros((pmt.shape[0], ff ** 2, ff ** 2)).astype(type_complex)

        for i, layer in enumerate(pmt):
            f_coeffs = fft_piecewise_constant(layer, fourier_order, type_complex=type_complex)
            center = ee.array(f_coeffs.shape) // 2

            conv_idx = ee.arange(-ff + 1, ff, 1)
            conv_idx = circulant(conv_idx)
            conv_i = ee.repeat(conv_idx, ff, 1)
            conv_i = ee.repeat(conv_i, ff, axis=0)
            conv_j = ee.tile(conv_idx, (ff, ff))
            e_conv = f_coeffs[center[0] + conv_i, center[1] + conv_j]
            res = res.at[i].set(e_conv)
            # res = res.at[i].set(f_coeffs[center[0] + conv_i, center[1] + conv_j])
            # assign_value = f_coeffs[center[0] + conv_i, center[1] + conv_j]
            # res = ee.assign(res, i, assign_value)

    # import matplotlib.pyplot as plt
    #
    # plt.figure()
    # plt.imshow(abs(res[0]), cmap='jet')
    # plt.colorbar()
    # plt.show()
    # print('conv time: ', time.time() - t0)
    return res


def to_conv_mat(pmt, fourier_order, type_complex=jnp.complex128):

    if len(pmt.shape) == 2:
        print('shape is 2')
        raise ValueError
    ff = 2 * fourier_order + 1

    if pmt.shape[1] == 1:  # 1D
        res = jnp.zeros((pmt.shape[0], ff, ff)).astype(type_complex)

        # extend array for FFT
        # minimum_pattern_size = (4 * fourier_order + 1) * pmt.shape[2]
        minimum_pattern_size = 2 * ff

        if pmt.shape[2] < minimum_pattern_size:
            n = minimum_pattern_size // pmt.shape[2]
            pmt = np.repeat(pmt, n+1, axis=2)

        for i, layer in enumerate(pmt):
            f_coeffs = jnp.fft.fftshift(jnp.fft.fft(layer / layer.size))
            center = f_coeffs.shape[1] // 2

            conv_idx = ee.arange(-ff + 1, ff, 1)
            conv_idx = circulant(conv_idx)
            e_conv = f_coeffs[0, center + conv_idx]
            res = res.at[i].set(e_conv)
            # res = ee.assign(res, i, e_conv)

    else:  # 2D
        # attention on the order of axis (Z Y X)
        res = jnp.zeros((pmt.shape[0], ff ** 2, ff ** 2)).astype(type_complex)

        # extend array
        minimum_pattern_size = 2 * ff
        if pmt.shape[1] < minimum_pattern_size:
            n = minimum_pattern_size // pmt.shape[1]
            pmt = jnp.repeat(pmt, n+1, axis=1)
        if pmt.shape[2] < minimum_pattern_size:
            n = minimum_pattern_size // pmt.shape[2]
            pmt = np.repeat(pmt, n+1, axis=2)

        for i, layer in enumerate(pmt):
            f_coeffs = jnp.fft.fftshift(jnp.fft.fft2(layer / layer.size))
            center = jnp.array(f_coeffs.shape) // 2

            conv_idx = jnp.arange(-ff + 1, ff, 1)
            conv_idx = circulant(conv_idx)

            conv_i = jnp.repeat(conv_idx, ff, 1)
            conv_i = jnp.repeat(conv_i, ff, axis=0)
            conv_j = jnp.tile(conv_idx, (ff, ff))
            e_conv = f_coeffs[center[0] + conv_i, center[1] + conv_j]
            res = res.at[i].set(e_conv)

            # res = res.at[i].set(f_coeffs[center[0] + conv_i, center[1] + conv_j])
            # res = ee.assign(res, i, assign_value)

    # import matplotlib.pyplot as plt
    #
    # plt.figure()
    # plt.imshow(abs(res[0]), cmap='jet')
    # plt.colorbar()
    # plt.show()
    # print('conv time: ', time.time() - t0)
    return res


def circulant(c):

    center = c.shape[0] // 2
    # circ = ee.zeros((center[0] + 1, center[0] + 1), dtype='int32')
    circ = ee.zeros((center + 1, center + 1), int)

    for r in range(center+1):
        idx = ee.arange(r, r - center - 1, -1)

        # circ = circ.at[r].set(c[center + idx])
        assign_value = c[center + idx]
        circ = ee.assign(circ, r, assign_value)

    return circ
