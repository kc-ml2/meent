import copy
# import numpy as np
import scipy.io
import jax.numpy as np

from scipy.linalg import circulant  # TODO: acceptable?

from pathlib import Path


def put_n_ridge_in_pattern(pattern_all, wl):

    pattern_all = copy.deepcopy(pattern_all)

    for i, (n_ridge, n_groove, pattern) in enumerate(pattern_all):

        if type(n_ridge) == str:
            material = n_ridge
            n_ridge = find_n_index(material, wl)
        pattern_all[i][0] = n_ridge
    return pattern_all


def find_n_index(material, wl):
    # TODO: where put this to?
    nk_path = str(Path(__file__).resolve().parent.parent) + '/nk_data/p_Si.mat'
    mat_si = scipy.io.loadmat(nk_path)

    mat_table = {}
    mat_table['SILICON'] = mat_si

    mat_property = mat_table[material.upper()]
    n_index = np.interp(wl, mat_property['WL'].flatten(), mat_property['n'].flatten())

    return n_index


def fill_factor_to_ucell(patterns_fill_factor, wl, grating_type):
    # from convolution_matrix import put_n_ridge_in_pattern, draw_fill_factor

    pattern_fill_factor = put_n_ridge_in_pattern(patterns_fill_factor, wl)
    ucell = draw_fill_factor(pattern_fill_factor, grating_type)

    return ucell


# def permittivity_mapping_by_fill_factor(pattern_all, wl, fourier_order, grating_type, oneover=False):
#     pattern_all = put_n_ridge_in_pattern(pattern_all, wl, oneover)
#
#     pmtvy = draw_fill_factor(pattern_all, grating_type)
#     conv_all = to_conv_mat_old(pmtvy, fourier_order)
#
#     return conv_all


def to_conv_mat_old(pmt, fourier_order):
    # FFT scaling: https://kr.mathworks.com/matlabcentral/answers/15770-scaling-the-fft-and-the-ifft?s_tid=srchtitle
    # pmt = zxy = np.swapaxes(pmt, 0, 2)
    if len(pmt.shape) == 2:
        print('shape is 2')
        raise ValueError

    # TODO: check whether 1D case is correct or not. I think I actually didn't test it at all.
    # if len(pmt.shape)==2 or pmt.shape[1] == 1:  # 1D
    if pmt.shape[1] == 1:  # 1D
        res = np.zeros((pmt.shape[0], 2*fourier_order+1, 2*fourier_order+1)).astype('complex')

        # extend array for FFT
        minimum_pattern_size = (4 * fourier_order + 1) * pmt.shape[2]  # TODO: what is theoretical minimum?
        if pmt.shape[2] < minimum_pattern_size:
            n = minimum_pattern_size // pmt.shape[2]
            pmt = np.repeat(pmt, n+1, axis=2)

        for i, pmtvy in enumerate(pmt):
            pmtvy_fft = np.fft.fftshift(np.fft.fftn(pmtvy / pmtvy.size))
            center = pmtvy_fft.shape[1] // 2
            pmtvy_fft_cut = (pmtvy_fft[0, -2*fourier_order + center: center + 2*fourier_order + 1])
            A = np.roll(circulant(pmtvy_fft_cut.flatten()), (pmtvy_fft_cut.size + 1) // 2, 0)
            # res[i] = A[:2*fourier_order+1, :2*fourier_order+1]
            #
            # exclude = [(2, 5), (3, 4), (6, 1)]
            #
            # ind = tuple(np.array(exclude).T)

            cut_idx = np.arange(2*fourier_order+1)
            xx, yy = np.meshgrid(cut_idx, cut_idx, indexing='ij')
            xx = xx.flatten()
            yy = yy.flatten()
            res = np.array([A[xx, yy].reshape((2*fourier_order+1, 2*fourier_order+1))])



    else:  # 2D
        # attention on the order of axis.
        # Here X Y Z. Cell Drawing in CAD is Y X Z. Here is Z Y X
        ff = 2 * fourier_order + 1

        # TODO: separate fourier order
        res = np.ndarray((pmt.shape[0], ff ** 2, ff ** 2)).astype('complex')

        # extend array
        # TODO: run test
        minimum_pattern_size = ff ** 2  # TODO: what is theoretical minimum?

        if pmt.shape[1] < minimum_pattern_size:
            n = minimum_pattern_size // pmt.shape[1]
            pmt = np.repeat(pmt, n+1, axis=1)
        if pmt.shape[2] < minimum_pattern_size:
            n = minimum_pattern_size // pmt.shape[2]
            pmt = np.repeat(pmt, n+1, axis=2)

        for i, layer in enumerate(pmt):
            pmtvy_fft = np.fft.fftshift(np.fft.fft2(layer / layer.size))

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
    #
    return res


# def cell_swap_axes(yxz):
#     zxy = np.swapaxes(yxz, 0, 2)
#     return zxy


# def to_conv_mat_3d(cell, fourier_order):
#     # attention on the order of axis.
#     # TODO: Here X Y Z. Cell Drawing in CAD is Y X Z.
#     ff = 2 * fourier_order + 1
#
#     # TODO: separate fourier order
#     res = np.ndarray((cell.shape[-1], ff ** 2, ff ** 2)).astype('complex')
#
#     # extend array
#     # TODO: run test
#     if cell.shape[0] < 2 * ff + 1:
#         n = (2 * ff + 1) // cell.shape[0]
#         cell = np.repeat(cell, n + 1, axis=0)
#     if cell.shape[1] < 2 * ff + 1:
#         n = (2 * ff + 1) // cell.shape[1]
#         cell = np.repeat(cell, n + 1, axis=1)
#
#     for i in range(cell.shape[-1]):
#         layer = cell[:, :, i]
#         pmtvy_fft = np.fft.fftshift(np.fft.fft2(layer / layer.size))
#
#         center = np.array(pmtvy_fft.shape) // 2
#
#         conv_idx = np.arange(ff - 1, -ff, -1)
#         conv_idx = circulant(conv_idx)[ff - 1:, :ff]
#
#         conv_i = np.repeat(conv_idx, ff, axis=1)
#         conv_i = np.repeat(conv_i, [ff] * ff, axis=0)
#         conv_j = np.tile(conv_idx, (ff, ff))
#         res[i] = pmtvy_fft[center[0] + conv_i, center[1] + conv_j]
#
#     return res


def draw_fill_factor(patterns_fill_factor, grating_type, resolution=1000):

    # res in Z X Y
    if grating_type == 2:
        res = np.ndarray((len(patterns_fill_factor), resolution, resolution))
    else:
        res = np.ndarray((len(patterns_fill_factor), 1, resolution))

    if grating_type in (0, 1):  # TODO: handle this by len(fill_factor)
        # fill_factor is not exactly implemented.
        for i, (n_ridge, n_groove, fill_factor) in enumerate(patterns_fill_factor):
            permittivity = np.ones((1, resolution))
            cut = int(resolution * fill_factor)
            permittivity[0, :cut] *= n_ridge ** 2
            permittivity[0, cut:] *= n_groove ** 2
            res[i, 0] = permittivity

    else:
        for i, (n_ridge, n_groove, fill_factor) in enumerate(patterns_fill_factor):
            fill_factor = np.array(fill_factor)
            permittivity = np.ones((resolution, resolution))
            cut = (resolution * fill_factor)
            permittivity *= n_groove ** 2
            permittivity[:int(cut[0]), :int(cut[1])] *= n_ridge ** 2
            res[i] = permittivity

    return res

