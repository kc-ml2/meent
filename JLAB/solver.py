import time
import numpy as np

from meent.on_numpy.rcwa import RCWALight as RCWA
from meent.on_numpy.convolution_matrix import put_n_ridge_in_pattern, to_conv_mat, find_n_index


class JLABCode(RCWA):
    def __init__(self, grating_type=0, n_I=1.45, n_II=1., theta=0, phi=0, psi=0, fourier_order=40, period=100,
                 wls=np.linspace(900, 900, 1), pol=1, patterns=None, ucell=None, thickness=(325,), algo='TMM'):

        super().__init__(0, grating_type, n_I, n_II, theta, phi, psi, fourier_order, period, wls, pol, patterns, ucell,
                         thickness, algo)

    # def permittivity_mapping_acs(self, wl):
    #     pattern = put_n_ridge_in_pattern(self.patterns, wl)
    #
    #     if self.grating_type == 2:
    #         resolutions = pattern[0][2].shape
    #         ucell = np.zeros((len(pattern), *resolutions), dtype='complex')
    #     else:
    #         resolution = len(pattern[0][2])
    #         ucell = np.zeros((len(pattern), 1, resolution), dtype='complex')
    #
    #     for i, (n_ridge, n_groove, pixel_map) in enumerate(pattern):
    #         pixel_map = np.array(pixel_map, dtype='complex')
    #         pixel_map = (pixel_map + 1) / 2
    #         pixel_map = pixel_map * (n_ridge ** 2 - n_groove ** 2) + n_groove ** 2
    #         ucell[i] = pixel_map
    #
    #     e_conv_all = to_conv_mat(ucell, self.fourier_order)
    #     o_e_conv_all = to_conv_mat(1 / ucell, self.fourier_order)
    #
    #     return e_conv_all, o_e_conv_all

    # def reproduce_acs(self, pattern_pixel):
    #     self.patterns = pattern_pixel
    #
    #     e_conv_all, o_e_conv_all = self.permittivity_mapping_acs(self.wls)
    #
    #     de_ri, de_ti = self.solve(self.wls, e_conv_all, o_e_conv_all)
    #     if self.grating_type == 0:
    #         center = de_ti.shape[0] // 2
    #         tran_cut = de_ti[center - 1:center + 2][::-1]
    #         refl_cut = de_ri[center - 1:center + 2][::-1]
    #     else:
    #         x_c, y_c = np.array(de_ti.shape) // 2
    #         tran_cut = de_ti[x_c - 1:x_c + 2, y_c - 1:y_c + 2][::-1, ::-1]
    #         refl_cut = de_ri[x_c - 1:x_c + 2, y_c - 1:y_c + 2][::-1, ::-1]
    #
    #     return tran_cut.flatten()[-1], refl_cut, tran_cut

    def reproduce_acs_cell(self, n_ridge, n_groove):

        if type(n_ridge) == str:
            n_ridge = find_n_index(n_ridge, self.wls)

        # self.ucell = np.array([[self.patterns]])

        self.ucell = (self.ucell + 1) / 2
        self.ucell = self.ucell * (n_ridge ** 2 - n_groove ** 2) + n_groove ** 2

        de_ri, de_ti = self.run_ucell()

        if self.grating_type == 0:
            center = de_ti.shape[0] // 2
            tran_cut = de_ti[center - 1:center + 2][::-1]
            refl_cut = de_ri[center - 1:center + 2][::-1]
        else:
            x_c, y_c = np.array(de_ti.shape) // 2
            tran_cut = de_ti[x_c - 1:x_c + 2, y_c - 1:y_c + 2][::-1, ::-1]
            refl_cut = de_ri[x_c - 1:x_c + 2, y_c - 1:y_c + 2][::-1, ::-1]

        return tran_cut.flatten()[-1], refl_cut, tran_cut

    # def reproduce_acs_loop_wavelength(self, pattern, trans_angle, wls=None):
    #     if wls is None:
    #         wls = self.wls
    #     else:
    #         self.wls = wls
    #     self.init_spectrum_array()
    #
    #     # self.patterns = [['SILICON', 1, pattern]]  # n_ridge, n_groove, pattern_pixel
    #     self.thickness = [325]
    #
    #     self.pol = 1
    #
    #     for i, wl in enumerate(wls):
    #         self.period = [abs(wl / np.sin(trans_angle / 180 * np.pi))]
    #
    #         e_conv_all, o_e_conv_all = self.permittivity_mapping_acs(wl)
    #
    #         de_ri, de_ti = self.solve_1d(wl, e_conv_all, o_e_conv_all)
    #
    #         self.save_spectrum_array(de_ri, de_ti, i)
    #
    #     return self.spectrum_r, self.spectrum_t

