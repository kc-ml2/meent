import time
import numpy as np

from meent.rcwa import RCWA
from meent.convolution_matrix import put_n_ridge_in_pattern, to_conv_mat_fill_factor


class JLABCode(RCWA):
    def __init__(self, grating_type=0, n_I=1.45, n_II=1., theta=0, phi=0, psi=0, fourier_order=40, period=100,
                 wls=np.linspace(900, 900, 1), polarization=1, patterns=None, thickness=(325,), algo='TMM'):

        super().__init__(grating_type, n_I, n_II, theta, phi, psi, fourier_order, period, wls, polarization, patterns,
                         thickness, algo)

    def permittivity_mapping_jlab(self, pattern_all, wl, period, fourier_order, oneover=False):
        pattern_all = put_n_ridge_in_pattern(self.patterns, wl, oneover)
        pmtvy = self.draw_1d_jlab(pattern_all)

        conv_all = to_conv_mat_fill_factor(pmtvy, fourier_order)

        return conv_all

    def draw_1d_jlab(self, patterns_pixel, resolution=1001):

        resolution = len(patterns_pixel[0][2].flatten())
        res = np.ndarray((len(patterns_pixel), resolution))

        for i, (n_ridge, n_groove, pixel_map) in enumerate(patterns_pixel):
            pixel_map = np.array(pixel_map, dtype='float')
            pixel_map = (pixel_map + 1) / 2
            pixel_map = pixel_map * (n_ridge ** 2 - n_groove ** 2) + n_groove ** 2
            res[i] = pixel_map

        return res

    def reproduce_acs(self, pattern_pixel, wavelength, trans_angle):
        self.n_I = 1.45  # glass
        self.n_II = 1

        self.theta = 0

        self.fourier_order = 40

        self.wls = np.array([wavelength])

        self.period = abs(self.wls / np.sin(trans_angle / 180 * np.pi))

        self.patterns = [['SILICON', 1, pattern_pixel]]  # n_ridge, n_groove, pattern_pixel
        self.thickness = [325]

        self.pol = 1

        E_conv_all = self.permittivity_mapping_jlab(pattern_pixel, self.wls, self.period, self.fourier_order)
        oneover_E_conv_all = self.permittivity_mapping_jlab(pattern_pixel, self.wls, self.period, self.fourier_order,
                                                            oneover=True)

        de_ri, de_ti = self.solve_1d(wavelength, E_conv_all, oneover_E_conv_all)

        center = de_ti.shape[0] // 2
        tran_cut = de_ti[center - 1:center + 2]

        return tran_cut[0], tran_cut


if __name__ == '__main__':
    t0 = time.perf_counter()
    wavelength = 900
    deflected_angle = 60
    algo = 'TMM'

    pattern = np.array([-1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                        -1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                        1., 1., 1., 1., 1., 1., 1., 1., -1., 1., 1., 1., 1.,
                        1., 1., 1., 1., 1., 1., -1., 1., 1., 1., 1., 1., 1.,
                        1., 1., 1., 1., 1., 1., 1., 1., 1., -1., 1., 1.])

    AA = JLABCode(algo=algo)
    a, b = AA.reproduce_acs(pattern, wavelength, deflected_angle)

    print('result:', a, b)
    print('time:', time.perf_counter() - t0)
