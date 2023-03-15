import time
from copy import deepcopy

import numpy as np

from ._base import _BaseRCWA
from .convolution_matrix import to_conv_mat_continuous, to_conv_mat_discrete, to_conv_mat_continuous_vector
from .field_distribution import field_dist_1d, field_dist_1d_conical, field_dist_2d, field_plot


class RCWANumpy(_BaseRCWA):
    def __init__(self,
                 n_I=1.,
                 n_II=1.,
                 theta=0,
                 phi=0,
                 psi=0,
                 period=(100, 100),
                 wavelength=900,
                 ucell=None,
                 ucell_info_list=None,
                 thickness=None,
                 mode=0,
                 grating_type=0,
                 pol=0,
                 fourier_order=2,
                 ucell_materials=None,
                 algo='TMM',
                 perturbation=1E-10,
                 device='cpu',
                 type_complex=np.complex128,
                 fft_type=0,
                 improve_dft=True,
                 **kwargs,
                 ):

        super().__init__(grating_type=grating_type, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi, pol=pol,
                         fourier_order=fourier_order, period=period, wavelength=wavelength,
                         thickness=thickness, algo=algo, perturbation=perturbation,
                         device=device, type_complex=type_complex, )

        self.ucell = deepcopy(ucell)
        self.ucell_materials = ucell_materials
        self.ucell_info_list = ucell_info_list

        self.mode = mode
        self.device = 'cpu'
        self.type_complex = type_complex
        self.fft_type = fft_type
        self.improve_dft = improve_dft

        # self.mat_table = read_material_table(type_complex=self.type_complex)
        self.layer_info_list = []

    def _solve(self, wavelength, e_conv_all, o_e_conv_all):
        self.kx_vector = self.get_kx_vector(wavelength)  # TODO: add ky_vector?

        if self.grating_type == 0:
            de_ri, de_ti, layer_info_list, T1 = self.solve_1d(wavelength, e_conv_all, o_e_conv_all)
        elif self.grating_type == 1:
            de_ri, de_ti, layer_info_list, T1 = self.solve_1d_conical(wavelength, e_conv_all, o_e_conv_all)
        elif self.grating_type == 2:
            de_ri, de_ti, layer_info_list, T1 = self.solve_2d(wavelength, e_conv_all, o_e_conv_all)
        else:
            raise ValueError

        return de_ri.real, de_ti.real, layer_info_list, T1, self.kx_vector

    def solve(self, wavelength, e_conv_all, o_e_conv_all):
        de_ri, de_ti, layer_info_list, T1, kx_vector = self._solve(wavelength, e_conv_all, o_e_conv_all)

        self.layer_info_list = layer_info_list
        self.T1 = T1
        self.kx_vector = kx_vector

        return de_ri, de_ti

    def conv_solve(self, *args, **kwargs):
        [setattr(self, k, v) for k, v in kwargs.items()]

        if self.fft_type == 0:
            E_conv_all = to_conv_mat_discrete(self.ucell, self.fourier_order, type_complex=self.type_complex,
                                              improve_dft=self.improve_dft)
            o_E_conv_all = to_conv_mat_discrete(1 / self.ucell, self.fourier_order, type_complex=self.type_complex,
                                                improve_dft=self.improve_dft)
        elif self.fft_type == 1:
            E_conv_all = to_conv_mat_continuous(self.ucell, self.fourier_order, type_complex=self.type_complex)
            o_E_conv_all = to_conv_mat_continuous(1 / self.ucell, self.fourier_order, type_complex=self.type_complex)
        elif self.fft_type == 2:
            E_conv_all, o_E_conv_all = to_conv_mat_continuous_vector(self.ucell_info_list, self.fourier_order,
                                                                     type_complex=self.type_complex)
        else:
            raise ValueError

        de_ri, de_ti, layer_info_list, T1, kx_vector = self._solve(self.wavelength, E_conv_all, o_E_conv_all)

        self.layer_info_list = layer_info_list
        self.T1 = T1
        self.kx_vector = kx_vector

        return de_ri, de_ti

    def calculate_field(self, resolution=None, plot=True):

        if self.grating_type == 0:
            resolution = [100, 1, 100] if not resolution else resolution
            field_cell = field_dist_1d(self.wavelength, self.kx_vector, self.n_I, self.theta, self.fourier_order,
                                       self.T1, self.layer_info_list, self.period, self.pol, resolution=resolution,
                                       type_complex=self.type_complex)
        elif self.grating_type == 1:
            resolution = [100, 1, 100] if not resolution else resolution
            field_cell = field_dist_1d_conical(self.wavelength, self.kx_vector, self.n_I, self.theta, self.phi,
                                               self.fourier_order, self.T1, self.layer_info_list, self.period,
                                               resolution=resolution, type_complex=self.type_complex)

        else:
            resolution = [10, 10, 10] if not resolution else resolution
            t0 = time.time()
            field_cell = field_dist_2d(self.wavelength, self.kx_vector, self.n_I, self.theta, self.phi,
                                       self.fourier_order, self.T1, self.layer_info_list, self.period,
                                       resolution=resolution, type_complex=self.type_complex)
        if plot:
            field_plot(field_cell, self.pol)

        return field_cell
