import time
import numpy as np

from meent.rcwa import RCWA
from meent.convolution_matrix import put_n_ridge_in_pattern, to_conv_mat_old


class JLABCode(RCWA):
    def __init__(self, grating_type=0, n_I=1.45, n_II=1., theta=0, phi=0, psi=0, fourier_order=40, period=100,
                 wls=np.linspace(900, 900, 1), polarization=1, patterns=None, thickness=(325,), algo='TMM'):

        super().__init__(grating_type, n_I, n_II, theta, phi, psi, fourier_order, period, wls, polarization, patterns,
                         thickness, algo)

    def permittivity_mapping_jlab(self, wl):
        pattern = put_n_ridge_in_pattern(self.patterns, wl)

        resolution = len(pattern[0][2])
        ucell = np.ndarray((len(pattern), 1, resolution))

        for i, (n_ridge, n_groove, pixel_map) in enumerate(pattern):
            pixel_map = np.array(pixel_map, dtype='float')
            pixel_map = (pixel_map + 1) / 2
            pixel_map = pixel_map * (n_ridge ** 2 - n_groove ** 2) + n_groove ** 2
            ucell[i] = pixel_map

        e_conv_all = to_conv_mat_old(ucell, self.fourier_order)
        o_e_conv_all = to_conv_mat_old(1/ucell, self.fourier_order)

        return e_conv_all, o_e_conv_all

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

        E_conv_all, oneover_E_conv_all = self.permittivity_mapping_jlab(self.wls)

        de_ri, de_ti = self.solve_1d(self.wls, E_conv_all, oneover_E_conv_all)  # TODO: check self.wls

        center = de_ti.shape[0] // 2
        tran_cut = de_ti[center - 1:center + 2]
        refl_cut = de_ri[center - 1:center + 2]

        return tran_cut[0], tran_cut, refl_cut

    def reproduce_acs_loop_wavelength(self, pattern, trans_angle, wls=None):
        if wls is None:
            wls = self.wls
        else:
            self.wls = wls
        self.init_spectrum_array()

        self.patterns = [['SILICON', 1, pattern]]  # n_ridge, n_groove, pattern_pixel
        self.thickness = [325]

        self.pol = 1

        for i, wl in enumerate(wls):
            self.period = [abs(wl / np.sin(trans_angle / 180 * np.pi))]

            E_conv_all, oneover_E_conv_all = self.permittivity_mapping_jlab(wl)

            de_ri, de_ti = self.solve_1d(wl, E_conv_all, oneover_E_conv_all)

            self.spectrum_r[i] = de_ri
            self.spectrum_t[i] = de_ti

        return self.spectrum_r, self.spectrum_t


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
    a, b, c = AA.reproduce_acs(pattern, wavelength, deflected_angle)

    print('result:', a, b, c)
    print('time:', time.perf_counter() - t0)
