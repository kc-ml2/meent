import time
import numpy as np

from ._base import _BaseRCWA
from .convolution_matrix import to_conv_mat, find_nk_index, fill_factor_to_ucell, put_permittivity_in_ucell, read_material_table


class RCWALight(_BaseRCWA):
    def __init__(self, mode=0, grating_type=0, n_I=1., n_II=1., theta=0, phi=0, psi=0, fourier_order=40, period=(100,),
                 wls=np.linspace(900, 900, 1), pol=0, patterns=None, ucell=None, ucell_materials=None, thickness=None, algo='TMM'):

        super().__init__(grating_type, n_I, n_II, theta, phi, psi, fourier_order, period, wls, pol, patterns, ucell, ucell_materials,
                         thickness, algo)

        self.mode = mode
        self.spectrum_r, self.spectrum_t = None, None
        self.init_spectrum_array()
        self.mat_table = read_material_table()


    def solve(self, wl, e_conv_all, o_e_conv_all):

        # TODO: !handle uniform layer

        if self.grating_type == 0:
            de_ri, de_ti = self.solve_1d(wl, e_conv_all, o_e_conv_all)
        elif self.grating_type == 1:
            de_ri, de_ti = self.solve_1d_conical(wl, e_conv_all, o_e_conv_all)
        elif self.grating_type == 2:
            de_ri, de_ti = self.solve_2d(wl, e_conv_all, o_e_conv_all)
        else:
            raise ValueError

        return de_ri.real, de_ti.real

    def loop_wavelength_fill_factor(self, wavelength_array=None):

        if wavelength_array is not None:
            self.wls = wavelength_array
            self.init_spectrum_array()

        for i, wl in enumerate(self.wls):

            ucell = fill_factor_to_ucell(self.patterns, wl, self.grating_type, self.mat_table)
            e_conv_all = to_conv_mat(ucell, self.fourier_order)
            o_e_conv_all = to_conv_mat(1 / ucell, self.fourier_order)

            de_ri, de_ti = self.solve(wl, e_conv_all, o_e_conv_all)
            self.spectrum_r[i] = de_ri
            self.spectrum_t[i] = de_ti

        return self.spectrum_r, self.spectrum_t

    # def loop_wavelength_ucell(self):
    #     wls = self.wls
    #     ucell = self.ucell
    #
    #     for i, wl in enumerate(self.wls):
    #         self.wls = np.array(wl)
    #         de_ri, de_ti = self.run_ucell()
    #
    #         self.spectrum_r[i] = de_ri
    #         self.spectrum_t[i] = de_ti
    #         self.ucell = ucell
    #
    #     self.wls = wls
    #
    #     return self.spectrum_r, self.spectrum_t

    def run_ucell(self):

        self.ucell = put_permittivity_in_ucell(self.ucell, self.ucell_materials, self.mat_table, self.wls)

        e_conv_all = to_conv_mat(self.ucell, self.fourier_order)
        o_e_conv_all = to_conv_mat(1 / self.ucell, self.fourier_order)

        de_ri, de_ti = self.solve(self.wls, e_conv_all, o_e_conv_all)

        return de_ri, de_ti


