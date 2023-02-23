from functools import partial

import jax

import jax.numpy as jnp
import numpy as np

from os import walk
from pathlib import Path
from scipy.io import loadmat


def put_permittivity_in_ucell(ucell, mat_list, mat_table, wl, type_complex=jnp.complex128):
    res = jnp.zeros(ucell.shape, dtype=type_complex)
    ucell_mask = jnp.array(ucell, dtype=type_complex)
    for i_mat, material in enumerate(mat_list):
        mask = jnp.nonzero(ucell_mask == i_mat)

        if type(material) == str:
            assign_value = find_nk_index(material, mat_table, wl) ** 2
        else:
            assign_value = type_complex(material ** 2)

        res = res.at[mask].set(assign_value)

    return res


def put_permittivity_in_ucell_old(ucell, mat_list, mat_table, wl, type_complex=jnp.complex128):

    res = jnp.zeros(ucell.shape, dtype=type_complex)

    for z in range(ucell.shape[0]):
        for y in range(ucell.shape[1]):
            for x in range(ucell.shape[2]):
                material = mat_list[int(ucell[z, y, x])]
                assign_index = (z, y, x)

                if type(material) == str:
                    assign_value = find_nk_index(material, mat_table, wl, type_complex=type_complex) ** 2
                else:
                    assign_value = type_complex(material ** 2)

                res = res.at[assign_index].set(assign_value)

    return res


def put_permittivity_in_ucell_object(ucell_size, mat_list, obj_list, mat_table, wl,
                                     type_complex=jnp.complex128):
    """
    under development
    """
    res = jnp.zeros(ucell_size, dtype=type_complex)

    for material, obj_index in zip(mat_list, obj_list):
        if type(material) == str:
            res[obj_index] = find_nk_index(material, mat_table, wl, type_complex=type_complex) ** 2
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
    nk = (n_index + 1j * k_index)

    return nk


def read_material_table(nk_path=None, type_complex=jnp.complex128):
    if type_complex == jnp.complex128:
        type_complex = jnp.float64
    elif type_complex == jnp.complex64:
        type_complex = jnp.float32
    else:
        raise ValueError

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
            mat_table[name[:-4].upper()] = type_complex(data)

        elif name[-3:] == 'mat':
            data = loadmat(path)
            data = jnp.array([type_complex(data['WL']), type_complex(data['n']), type_complex(data['k'])])[:, :, 0].T
            mat_table[name[:-4].upper()] = data
    return mat_table


def cell_compression(cell, type_complex=jnp.complex128):

    if type_complex == jnp.complex128:
        type_float = jnp.float64
    else:
        type_float = jnp.float32

    # find discontinuities in x
    step_y, step_x = 1. / jnp.array(cell.shape, dtype=type_float)
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


# @partial(jax.jit, static_argnums=(1,2 ))
def fft_piecewise_constant(cell, fourier_order, type_complex=jnp.complex128):

    if cell.shape[0] == 1:
        fourier_order = [0, fourier_order]
    else:
        fourier_order = [fourier_order, fourier_order]
    cell, x, y = cell_compression(cell, type_complex=type_complex)

    # X axis
    cell_next_x = jnp.roll(cell, -1, axis=1)
    cell_diff_x = cell_next_x - cell

    modes = jnp.arange(-2 * fourier_order[1], 2 * fourier_order[1] + 1, 1)

    f_coeffs_x = cell_diff_x @ jnp.exp(-1j * 2 * jnp.pi * x @ modes[None, :]).astype(type_complex)
    c = f_coeffs_x.shape[1] // 2

    x_next = jnp.vstack((jnp.roll(x, -1, axis=0)[:-1], 1)) - x

    assign_index = (jnp.arange(len(f_coeffs_x)), jnp.array([c]))
    assign_value = (cell @ jnp.vstack((x[0], x_next[:-1]))).flatten().astype(type_complex)
    f_coeffs_x = f_coeffs_x.at[assign_index].set(assign_value)

    mask_int = jnp.hstack([jnp.arange(c), jnp.arange(c+1, f_coeffs_x.shape[1])])
    assign_index = mask_int
    assign_value = f_coeffs_x[:, mask_int] / (-1j * 2 * jnp.pi * modes[mask_int])
    f_coeffs_x = f_coeffs_x.at[:, assign_index].set(assign_value)

    # Y axis
    f_coeffs_x_next_y = jnp.roll(f_coeffs_x, -1, axis=0)
    f_coeffs_x_diff_y = f_coeffs_x_next_y - f_coeffs_x

    modes = jnp.arange(-2 * fourier_order[0], 2 * fourier_order[0] + 1, 1)

    f_coeffs_xy = f_coeffs_x_diff_y.T @ jnp.exp(-1j * 2 * jnp.pi * y @ modes[None, :]).astype(type_complex)
    c = f_coeffs_xy.shape[1] // 2

    y_next = jnp.vstack((jnp.roll(y, -1, axis=0)[:-1], 1)) - y

    assign_index = [c]
    assign_value = f_coeffs_x.T @ jnp.vstack((y[0], y_next[:-1])).astype(type_complex)
    f_coeffs_xy = f_coeffs_xy.at[:, assign_index].set(assign_value)

    if c:
        mask_int = jnp.hstack([jnp.arange(c), jnp.arange(c + 1, f_coeffs_x.shape[1])])

        assign_index = mask_int
        assign_value = f_coeffs_xy[:, mask_int] / (-1j * 2 * jnp.pi * modes[mask_int])

        f_coeffs_xy = f_coeffs_xy.at[:, assign_index].set(assign_value)

    return f_coeffs_xy.T


def to_conv_mat_continuous(pmt, fourier_order, device=None, type_complex=jnp.complex128):

    if len(pmt.shape) == 2:
        print('shape is 2')
        raise ValueError
    ff = 2 * fourier_order + 1

    if pmt.shape[1] == 1:  # 1D
        res = jnp.zeros((pmt.shape[0], ff, ff)).astype(type_complex)

        for i, layer in enumerate(pmt):
            f_coeffs = fft_piecewise_constant(layer, fourier_order, type_complex=type_complex)
            center = f_coeffs.shape[1] // 2
            conv_idx = jnp.arange(-ff + 1, ff, 1)
            conv_idx = circulant(conv_idx)
            e_conv = f_coeffs[0, center + conv_idx]
            res = res.at[i].set(e_conv)

    else:  # 2D
        # attention on the order of axis (Z Y X)
        res = jnp.zeros((pmt.shape[0], ff ** 2, ff ** 2)).astype(type_complex)

        for i, layer in enumerate(pmt):
            f_coeffs = fft_piecewise_constant(layer, fourier_order, type_complex=type_complex)
            center = jnp.array(f_coeffs.shape) // 2

            conv_idx = jnp.arange(-ff + 1, ff, 1)
            conv_idx = circulant(conv_idx)
            conv_i = jnp.repeat(conv_idx, ff, 1)
            conv_i = jnp.repeat(conv_i, ff, axis=0)
            conv_j = jnp.tile(conv_idx, (ff, ff))
            e_conv = f_coeffs[center[0] + conv_i, center[1] + conv_j]
            res = res.at[i].set(e_conv)

    # import matplotlib.pyplot as plt
    #
    # plt.figure()
    # plt.imshow(abs(res[0]), cmap='jet')
    # plt.colorbar()
    # plt.show()
    # print('conv time: ', time.time() - t0)
    return res


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def to_conv_mat_discrete(pmt, fourier_order, device=None, type_complex=jnp.complex128, improve_dft=True):

    if len(pmt.shape) == 2:
        print('shape is 2')
        raise ValueError
    ff = 2 * fourier_order + 1

    if pmt.shape[1] == 1:  # 1D
        res = jnp.zeros((pmt.shape[0], ff, ff)).astype(type_complex)
        minimum_pattern_size = (4 * fourier_order + 1) * pmt.shape[2]

        for i, layer in enumerate(pmt):
            if improve_dft and layer.shape[1] < minimum_pattern_size:  # extend array
                n = minimum_pattern_size // layer.shape[1]
                layer = np.repeat(layer, n + 1, axis=1)

            f_coeffs = jnp.fft.fftshift(jnp.fft.fft(layer / layer.size))
            center = f_coeffs.shape[1] // 2

            conv_idx = jnp.arange(-ff + 1, ff, 1)
            conv_idx = circulant(conv_idx)
            e_conv = f_coeffs[0, center + conv_idx]
            res = res.at[i].set(e_conv)

    else:  # 2D
        # attention on the order of axis (Z Y X)
        res = jnp.zeros((pmt.shape[0], ff ** 2, ff ** 2)).astype(type_complex)
        minimum_pattern_size_1 = 2 * ff * pmt.shape[1]
        minimum_pattern_size_2 = 2 * ff * pmt.shape[2]
        # 9 * (40*500) * (40*500) / 1E6 = 3600 MB = 3.6 GB

        for i, layer in enumerate(pmt):
            if improve_dft:  # extend array
                if layer.shape[0] < minimum_pattern_size_1:
                    n = minimum_pattern_size_1 // layer.shape[0]
                    layer = jnp.repeat(layer, n + 1, axis=0)
                if layer.shape[1] < minimum_pattern_size_2:
                    n = minimum_pattern_size_2 // layer.shape[1]
                    layer = jnp.repeat(layer, n + 1, axis=1)

            f_coeffs = jnp.fft.fftshift(jnp.fft.fft2(layer / layer.size))
            center = jnp.array(f_coeffs.shape) // 2

            conv_idx = jnp.arange(-ff + 1, ff, 1)
            conv_idx = circulant(conv_idx)

            conv_i = jnp.repeat(conv_idx, ff, 1)
            conv_i = jnp.repeat(conv_i, ff, axis=0)
            conv_j = jnp.tile(conv_idx, (ff, ff))
            e_conv = f_coeffs[center[0] + conv_i, center[1] + conv_j]
            res = res.at[i].set(e_conv)

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
    circ = jnp.zeros((center + 1, center + 1), int)

    for r in range(center+1):
        idx = jnp.arange(r, r - center - 1, -1)
        circ = circ.at[r].set(c[center + idx])

    return circ
