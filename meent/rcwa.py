import time
import numpy as np

from meent._base import _BaseRCWA
from meent.convolution_matrix import to_conv_mat_3d, find_n_index


class RCWA(_BaseRCWA):
    def __init__(self, grating_type=0, n_I=1., n_II=1., theta=0, phi=0, psi=0, fourier_order=40, period=(100,),
                 wls=np.linspace(900, 900, 1), pol=1, patterns=None, thickness=(325,), algo='TMM'):

        super().__init__(grating_type, n_I, n_II, theta, phi, psi, fourier_order, period, wls, pol, patterns,
                         thickness, algo)
        self.spectrum_r, self.spectrum_t = None, None
        self.init_spectrum_array()

    def solve(self, wl, E_conv_all, oneover_E_conv_all):

        if self.grating_type == 0:
            de_ri, de_ti = self.solve_1d(wl, E_conv_all, oneover_E_conv_all)
        elif self.grating_type == 1:
            # de_ri, de_ti = self.solve_1d_conical()  # TODO: implement
            de_ri = de_ti = None
        elif self.grating_type == 2:
            de_ri, de_ti = self.solve_2d(wl, E_conv_all, oneover_E_conv_all)
        else:
            raise ValueError

        return de_ri, de_ti

    def loop_wavelength_fill_factor(self, wavelength_array=None):

        if wavelength_array is not None:
            self.wls = wavelength_array
            self.init_spectrum_array()

        for i, wl in enumerate(self.wls):
            e_conv_all, oneover_e_conv_all = self.get_e_conv_set_by_fill_factor(wl)

            de_ri, de_ti = self.solve(wl, e_conv_all, oneover_e_conv_all)

            self.spectrum_r[i] = de_ri
            self.spectrum_t[i] = de_ti

        return self.spectrum_r, self.spectrum_t

    def loop_wavelength_ucell(self):

        cell = np.zeros((4, 4, 4))

        # si = [[0, 4], [0, 4], [0, 1]]
        # oxide = [[0, 4], [0, 4], [1, 2]]
        # ni = [[0, 4], [0, 4], [2, 3]]
        # ti = [[0, 4], [0, 4], [3, 4]]

        # si = [[x_begin, x_end], [y_begin, y_end], [z_begin, z_end]]
        si = ['SILICON', 0, 4, 0, 4, 0, 1]
        ox = ['SILICON', 0, 4, 0, 4, 1, 2]
        ni = ['SILICON', 0, 4, 0, 4, 2, 3]
        ti = ['SILICON', 0, 4, 0, 4, 3, 4]

        for i, wl in enumerate(self.wls):
            for material, x_begin, x_end, y_begin, y_end, z_begin, z_end in [si, ox, ni, ti]:
                n_index = find_n_index(material, wl)
                cell[x_begin:x_end, y_begin:y_end, z_begin:z_end] = n_index ** 2

            e_conv_all = to_conv_mat_3d(cell, fourier_order)
            oneover_e_conv_all = to_conv_mat_3d(1 / cell, fourier_order)

            de_ri, de_ti = self.solve(wl, e_conv_all, oneover_e_conv_all)

            self.spectrum_r[i] = de_ri
            self.spectrum_t[i] = de_ti

        return self.spectrum_r, self.spectrum_t


if __name__ == '__main__':
    grating_type = 2
    pol = 0

    n_I = 1
    n_II = 1

    theta = 0
    phi = 0
    psi = 0 if pol else 90

    wls = np.linspace(500, 2300, 100)

    if grating_type == 0:
        period = [700]
        patterns = [[3.48, 1, 0.3], [3.48, 1, 0.3]]  # n_ridge, n_groove, fill_factor
        fourier_order = 40

    elif grating_type == 2:
        period = [700, 700]
        patterns = [[3.48, 1, [0.3, 1]], [3.48, 1, [0.3, 1]]]  # n_ridge, n_groove, fill_factor
        fourier_order = 2
    else:
        raise ValueError

    thickness = [460, 660]

    AA = RCWA(grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
              fourier_order=fourier_order, wls=wls, period=period, patterns=patterns, thickness=thickness)
    t0 = time.perf_counter()

    a, b = AA.loop_wavelength_fill_factor()
    AA.plot()

    print(time.perf_counter() - t0)

    AA.loop_wavelength_ucell()
    AA.plot()
    print(1)
