import copy
import numpy as np

from scipy.io import loadmat
from scipy.linalg import circulant
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
    nk_path = str(Path(__file__).resolve().parent.parent.parent) + '/nk_data/p_Si.mat'  # TODO: organize
    mat_si = loadmat(nk_path)

    mat_table = {}
    mat_table['SILICON'] = mat_si

    mat_property = mat_table[material.upper()]
    n_index = np.interp(wl, mat_property['WL'].flatten(), mat_property['n'].flatten())

    return n_index


def fill_factor_to_ucell(patterns_fill_factor, wl, grating_type):
    pattern_fill_factor = put_n_ridge_in_pattern(patterns_fill_factor, wl)
    ucell = draw_fill_factor(pattern_fill_factor, grating_type)

    return ucell


def to_conv_mat(pmt, fourier_order):
    # FFT scaling: https://kr.mathworks.com/matlabcentral/answers/15770-scaling-the-fft-and-the-ifft?s_tid=srchtitle
    if len(pmt.shape) == 2:
        print('shape is 2')
        raise ValueError
    ff = 2 * fourier_order + 1

    # if len(pmt.shape)==2 or pmt.shape[1] == 1:  # 1D
    if pmt.shape[1] == 1:  # 1D  # TODO: confirm this handles all cases
        res = np.zeros((pmt.shape[0], 2*fourier_order+1, 2*fourier_order+1)).astype('complex')

        # extend array for FFT
        minimum_pattern_size = (4 * fourier_order + 1) * pmt.shape[2]
        # TODO: what is theoretical minimum?
        # TODO: can be a scalability issue
        if pmt.shape[2] < minimum_pattern_size:
            n = minimum_pattern_size // pmt.shape[2]
            pmt = np.repeat(pmt, n+1, axis=2)

        for i, pmtvy in enumerate(pmt):
            pmtvy_fft = np.fft.fftshift(np.fft.fftn(pmtvy / pmtvy.size))
            center = pmtvy_fft.shape[1] // 2

            pmtvy_fft_cut = (pmtvy_fft[0, -2*fourier_order + center: center + 2*fourier_order + 1])
            A = np.roll(circulant(pmtvy_fft_cut.flatten()), (pmtvy_fft_cut.size + 1) // 2, 0)
            res[i] = A[:2*fourier_order+1, :2*fourier_order+1]

    else:  # 2D
        # attention on the order of axis.
        # Here X Y Z. Cell Drawing in CAD is Y X Z. Here is Z Y X

        # TODO: separate fourier order
        res = np.zeros((pmt.shape[0], ff ** 2, ff ** 2)).astype('complex')

        # extend array
        # TODO: run test
        minimum_pattern_size = ff ** 2
        # TODO: what is theoretical minimum?
        # TODO: can be a scalability issue

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
