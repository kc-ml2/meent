import copy
import numpy as np
import scipy.io

from scipy.linalg import circulant

from pathlib import Path


def put_n_ridge_in_pattern(pattern_all, wl, oneover=False):
    # nk_path = str(Path(__file__).resolve().parent.parent) + '/nk_data/p_Si.mat'
    # mat_si = scipy.io.loadmat(nk_path)
    #
    # materials = {}
    # materials['SILICON'] = mat_si

    pattern_all = copy.deepcopy(pattern_all)

    for i, (n_ridge, n_groove, pattern) in enumerate(pattern_all):

        if type(n_ridge) == str:
            material = n_ridge
            n_ridge = find_n_index(material, wl)
        pattern_all[i][0] = n_ridge if not oneover else 1 / n_ridge
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


def permittivity_mapping_by_fill_factor(pattern_all, wl, period, fourier_order, oneover=False):
    pattern_all = put_n_ridge_in_pattern(pattern_all, wl, oneover)
    if len(period) == 1:
        pmtvy = draw_1d_fill_factor(pattern_all)
    else:
        pmtvy = draw_2d_fill_factor(pattern_all)

    conv_all = to_conv_mat_fill_factor(pmtvy, fourier_order)

    return conv_all


def to_conv_mat_fill_factor(pmt, fourier_order):
    # FFT scaling: https://kr.mathworks.com/matlabcentral/answers/15770-scaling-the-fft-and-the-ifft?s_tid=srchtitle
    ff = 2 * fourier_order + 1

    # TODO: check whether 1D case is correct or not. I think I actually didn't test it at all.
    if len(pmt[0].shape) == 1:  # 1D
        res = np.ndarray((len(pmt), 2*fourier_order+1, 2*fourier_order+1)).astype('complex')

        # extend array for FFT
        if pmt.shape[1] < 2 * ff + 1:
            n = (2 * ff + 1) // pmt.shape[1]
            pmt = np.repeat(pmt, n+1, axis=1)

        for i, pmtvy in enumerate(pmt):
            pmtvy_fft = np.fft.fftshift(np.fft.fftn(pmtvy / pmtvy.size))

            center = len(pmtvy_fft) // 2
            pmtvy_fft_cut = (pmtvy_fft[-ff + center: center+ff+1])
            A = np.roll(circulant(pmtvy_fft_cut.flatten()), (pmtvy_fft_cut.size + 1) // 2, 0)
            res[i] = A[:2*fourier_order+1, :2*fourier_order+1]
            # res[i] = circulant(pmtvy_fft_cut)

    else:  # 2D
        # TODO: separate fourier order
        res = np.ndarray((len(pmt), ff ** 2, ff ** 2)).astype('complex')

        # extend array
        # TODO: run test
        if pmt.shape[1] < 2 * ff + 1:
            n = (2 * ff + 1) // pmt.shape[1]  # TODO: shape[1]? or [0]?
            pmt = np.repeat(pmt, n+1, axis=1)
        if pmt.shape[2] < 2 * ff + 1:
            n = (2 * ff + 1) // pmt.shape[2]
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


def to_conv_mat_3d(cell, fourier_order):
    # attention on the order of axis.
    # TODO: Here X Y Z. Cell Drawing in CAD is Y X Z.
    ff = 2 * fourier_order + 1

    # TODO: separate fourier order
    res = np.ndarray((cell.shape[-1], ff ** 2, ff ** 2)).astype('complex')

    # extend array
    # TODO: run test
    if cell.shape[0] < 2 * ff + 1:
        n = (2 * ff + 1) // cell.shape[0]
        cell = np.repeat(cell, n + 1, axis=0)
    if cell.shape[1] < 2 * ff + 1:
        n = (2 * ff + 1) // cell.shape[1]
        cell = np.repeat(cell, n + 1, axis=1)

    for i in range(cell.shape[-1]):
        layer = cell[:, :, i]
        pmtvy_fft = np.fft.fftshift(np.fft.fft2(layer / layer.size))

        center = np.array(pmtvy_fft.shape) // 2

        conv_idx = np.arange(ff - 1, -ff, -1)
        conv_idx = circulant(conv_idx)[ff - 1:, :ff]

        conv_i = np.repeat(conv_idx, ff, axis=1)
        conv_i = np.repeat(conv_i, [ff] * ff, axis=0)
        conv_j = np.tile(conv_idx, (ff, ff))
        res[i] = pmtvy_fft[center[0] + conv_i, center[1] + conv_j]

    return res


def draw_1d_fill_factor(patterns, resolution=1001):
    # fill_factor is not exactly implemented.
    res = np.ndarray((len(patterns), resolution))

    for i, (n_ridge, n_groove, fill_factor) in enumerate(patterns):

        permittivity = np.ones(resolution)
        cut = int(resolution * fill_factor)
        permittivity[:cut] *= n_ridge ** 2
        permittivity[cut:] *= n_groove ** 2
        res[i] = permittivity

    return res


def draw_2d(patterns, resolution=1001):
    # TODO: Implement
    res = np.ndarray((len(patterns), resolution, resolution))

    return res


def draw_2d_fill_factor(patterns, resolution=1001):
    res = np.ndarray((len(patterns), resolution, resolution))

    for i, (n_ridge, n_groove, fill_factor) in enumerate(patterns):
        fill_factor = np.array(fill_factor).reshape(-1)  # TODO: handle outside?
        permittivity = np.ones((resolution, resolution))
        cut = (resolution * fill_factor)
        permittivity *= n_groove ** 2
        permittivity[:int(cut[-1]), :int(cut[0])] *= n_ridge ** 2
        res[i] = permittivity

    return res
