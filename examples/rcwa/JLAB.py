import time
import numpy as np

from meent.rcwa import RCWA


class JLABCode(RCWA):
    def __init__(self, grating_type=0, n_I=1.45, n_II=1., theta=0, phi=0, psi=0, fourier_order=40, period=100,
                 wls=np.linspace(900, 900, 1), polarization=1, patterns=None, thickness=(325,), algo='TMM'):

        super().__init__(grating_type, n_I, n_II, theta, phi, psi, fourier_order, period, wls, polarization, patterns,
                         thickness, algo)

    def reproduce_acs(self, pattern_pixel, wavelength, trans_angle, algo):
        self.n_I = 1.45  # glass
        self.n_II = 1

        self.theta = 0
        self.phi = 0
        self.psi = 0

        self.fourier_order = 40

        self.wls = np.array([wavelength])

        self.period = abs(self.wls / np.sin(trans_angle / 180 * np.pi))

        # permittivity in grating layer
        self.patterns = [['SILICON', 1, pattern_pixel]]  # n_ridge, n_groove, pattern_pixel

        self.thickness = [325]

        self.pol = 1

        self.algo = algo

        if len(self.wls) != 1:
            raise ValueError('wavelength should be a single value')

        E_conv_all, oneover_E_conv_all = self.get_permittivity_map(wavelength)

        de_ri, de_ti = self.lalanne_1d(wavelength, E_conv_all, oneover_E_conv_all)

        center = de_ti.shape[0] // 2
        tran_cut = de_ti[center - 1:center + 2]

        return tran_cut[0], tran_cut


if __name__ == '__main__':
    AA = JLABCode()
    t0 = time.perf_counter()

    pattern = np.array([-1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                        -1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                        1., 1., 1., 1., 1., 1., 1., 1., -1., 1., 1., 1., 1.,
                        1., 1., 1., 1., 1., 1., -1., 1., 1., 1., 1., 1., 1.,
                        1., 1., 1., 1., 1., 1., 1., 1., 1., -1., 1., 1.])
    a, b = AA.reproduce_acs(pattern, 900, 60, 'TMM')
    print('result:', a, b)
    print('time:', time.perf_counter() - t0)
