import copy
import numpy as np
import scipy.io

from scipy.linalg import circulant

from pathlib import Path


def permittivity_mapping(patterns, wl, period, fourier_order, oneover=False):
    # TODO: not fully implemented

    # nk_pool = {'SILICON': np.array([[100, 400, 500, 600, 900], [3.48, 3.48, 3.48, 3.48, 3.48]])}

    nk_path = str(Path(__file__).resolve().parent.parent) + '/nk_data/p_Si.mat'
    mat_si = scipy.io.loadmat(nk_path)

    patterns = copy.deepcopy(patterns)

    for i, layer in enumerate(patterns):
        n_ridge = np.interp(wl, mat_si['WL'].flatten(), mat_si['n'].flatten())
        n_ridge = 3.48  # TODO: Hardcoding for test
        patterns[i][0] = n_ridge if not oneover else 1 / n_ridge

    if type(period) in [float, int] or len(period) == 1:
        pmtvy = draw_1d(patterns)
        # pmtvy = draw_1d_jlab(patterns)

    else:
        pmtvy = draw_2d(patterns)
    conv_all = to_conv_mat(pmtvy, fourier_order)

    return conv_all


def to_conv_mat(pmt, fourier_order):
    # FFT scaling
    # https://kr.mathworks.com/matlabcentral/answers/15770-scaling-the-fft-and-the-ifft#:~:text=the%20matlab%20fft%20outputs%202,point%20is%20the%20parseval%20equation.
    ff = 2 * fourier_order + 1

    # TODO: check whether 1D case is correct or not. I think I actually didn't test it at all.
    if len(pmt[0].shape) == 1:  # 1D
        res = np.ndarray((len(pmt), 2*fourier_order+1, 2*fourier_order+1)).astype('complex')

        # extend array
        if pmt.shape[1] < 2 * ff + 1:
            n = (2 * ff + 1) // pmt.shape[1]
            pmt = np.repeat(pmt, n+1, axis=1)

        for i, pmtvy in enumerate(pmt):
            pmtvy_fft = np.fft.fftn(pmtvy / pmtvy.size)
            pmtvy_fft = np.fft.fftshift(pmtvy_fft)

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
        if pmt.shape[0] < 2 * ff + 1:
            n = (2 * ff + 1) // pmt.shape[1]
            permittivities = np.repeat(pmt, n+1, axis=0)
        if pmt.shape[1] < 2 * ff + 1:
            n = (2 * ff + 1) // pmt.shape[1]
            pmt = np.repeat(pmt, n+1, axis=1)

        for i, layer in enumerate(pmt):
            pmtvy_fft = np.fft.fftn(layer / layer.size)
            pmtvy_fft = np.fft.fftshift(pmtvy_fft)

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
    # #
    return res


def draw_1d(patterns, resolution=1001):
    # fill_factor is not exactly implemented.
    res = np.ndarray((len(patterns), resolution))

    for i, (n_ridge, n_groove, fill_factor) in enumerate(patterns):

        permittivity = np.ones(resolution)
        cut = int(resolution * fill_factor)
        permittivity[:cut] *= n_ridge ** 2
        permittivity[cut:] *= n_groove ** 2
        res[i] = permittivity

    return res


def draw_1d_jlab(patterns_pixel, resolution=1001):

    resolution = len(patterns_pixel[0][2].flatten())
    res = np.ndarray((len(patterns_pixel), resolution))

    for i, (n_ridge, n_groove, pixel_map) in enumerate(patterns_pixel):
        pixel_map = np.array(pixel_map, dtype='float')
        # permittivity = np.ones(resolution) * n_groove
        pixel_map = (pixel_map + 1) / 2
        pixel_map = pixel_map * (n_ridge**2 - n_groove**2) + n_groove ** 2
        res[i] = pixel_map

    return res


def draw_2d(patterns, resolution=1001):

    # TODO: Implement
    res = np.ndarray((len(patterns), resolution, resolution))

    for i, (n_ridge, n_groove, fill_factor) in enumerate(patterns):

        permittivity = np.ones((resolution, resolution))
        cut = int(resolution * fill_factor)
        permittivity[:, :] *= n_groove ** 2
        permittivity[:, :cut] *= n_ridge ** 2
        res[i] = permittivity

    return res
