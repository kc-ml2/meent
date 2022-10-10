import time
import numpy as np

from meent.on_numpy.rcwa import RCWALight as RCWA
from meent.on_numpy.convolution_matrix import to_conv_mat, find_nk_index


class JLABCode(RCWA):
    def __init__(self, grating_type=0, n_I=1.45, n_II=1., theta=0, phi=0, psi=0, fourier_order=40, period=100,
                 wls=np.linspace(900, 900, 1), pol=1, patterns=None, ucell=None, ucell_materials=None, thickness=(325,), algo='TMM'):

        super().__init__(0, grating_type, n_I, n_II, theta, phi, psi, fourier_order, period, wls, pol, patterns, ucell,
                         ucell_materials,
                         thickness, algo)

    def reproduce_acs_cell(self, n_ridge, n_groove):

        if type(n_ridge) == str:
            n_ridge = find_nk_index(n_ridge, self.mat_table, self.wls)

        self.ucell = (self.ucell + 1) / 2
        self.ucell = self.ucell * (n_ridge ** 2 - n_groove ** 2) + n_groove ** 2

        e_conv_all = to_conv_mat(self.ucell, self.fourier_order)
        o_e_conv_all = to_conv_mat(1 / self.ucell, self.fourier_order)

        de_ri, de_ti = self.solve(self.wls, e_conv_all, o_e_conv_all)

        if self.grating_type == 0:
            center = de_ti.shape[0] // 2
            tran_cut = de_ti[center - 1:center + 2][::-1]
            refl_cut = de_ri[center - 1:center + 2][::-1]
        else:
            x_c, y_c = np.array(de_ti.shape) // 2
            tran_cut = de_ti[x_c - 1:x_c + 2, y_c - 1:y_c + 2][::-1, ::-1]
            refl_cut = de_ri[x_c - 1:x_c + 2, y_c - 1:y_c + 2][::-1, ::-1]

        return tran_cut.flatten()[-1], refl_cut, tran_cut
