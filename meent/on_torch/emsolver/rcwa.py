import time
import torch

import numpy as np

from ._base import _BaseRCWA
from .convolution_matrix import to_conv_mat_discrete, to_conv_mat_continuous, to_conv_mat_continuous_vector
from .field_distribution import field_dist_1d_vanilla, field_dist_2d_vanilla, field_plot, field_dist_1d_conical_vanilla, \
    field_dist_1d_vectorized_ji, field_dist_1d_vectorized_kji, field_dist_1d_conical_vectorized_ji, \
    field_dist_1d_conical_vectorized_kji, field_dist_2d_vectorized_ji, field_dist_2d_vectorized_kji


class RCWATorch(_BaseRCWA):
    def __init__(self,
                 n_I=1.,
                 n_II=1.,
                 theta=0.,
                 phi=0.,
                 period=(100., 100.),
                 wavelength=900.,
                 ucell=None,
                 ucell_info_list=None,
                 thickness=(0., ),
                 backend=2,
                 grating_type=0,
                 pol=0.,
                 fourier_order=(2, 0),
                 ucell_materials=None,
                 algo='TMM',
                 perturbation=1E-20,
                 device='cpu',
                 type_complex=torch.complex128,
                 fft_type=0,
                 improve_dft=True,
                 **kwargs,
                 ):

        super().__init__(grating_type=grating_type, n_I=n_I, n_II=n_II, theta=theta, phi=phi, pol=pol,
                         fourier_order=fourier_order, period=period, wavelength=wavelength,
                         thickness=thickness, algo=algo, perturbation=perturbation,
                         device=device, type_complex=type_complex)

        self.ucell = ucell
        self.ucell_materials = ucell_materials
        self.ucell_info_list = ucell_info_list

        self.backend = backend
        self.fft_type = fft_type
        self.improve_dft = improve_dft

        self.layer_info_list = []

    @property
    def ucell(self):
        return self._ucell

    @ucell.setter
    def ucell(self, ucell):
        if type(ucell) is torch.Tensor:
            if ucell.dtype in (torch.complex128, torch.complex64):
                dtype = self.type_complex
            elif ucell.dtype in (torch.float64, torch.float32, torch.int64, torch.int32):
                dtype = self.type_float
            else:
                raise ValueError
            self._ucell = ucell.to(device=self.device, dtype=dtype)
        elif isinstance(ucell, np.ndarray):
            if ucell.dtype in (np.int64, np.float64, np.int32, np.float32):
                dtype = self.type_float
            elif ucell.dtype in (np.complex128, np.complex64):
                dtype = self.type_complex
            else:
                raise ValueError
            self._ucell = torch.tensor(ucell, device=self.device, dtype=dtype)
        elif ucell is None:
            self._ucell = ucell
        else:
            raise ValueError

    def _solve(self, wavelength, e_conv_all, o_e_conv_all):
        self.kx_vector = self.get_kx_vector(wavelength)

        if self.grating_type == 0:
            de_ri, de_ti, layer_info_list, T1 = self.solve_1d(wavelength, e_conv_all, o_e_conv_all)
        elif self.grating_type == 1:
            de_ri, de_ti, layer_info_list, T1 = self.solve_1d_conical(wavelength, e_conv_all, o_e_conv_all)
        elif self.grating_type == 2:
            de_ri, de_ti, layer_info_list, T1 = self.solve_2d(wavelength, e_conv_all, o_e_conv_all)
        else:
            raise ValueError

        return de_ri, de_ti, layer_info_list, T1, self.kx_vector

    def solve(self, wavelength, e_conv_all, o_e_conv_all):
        de_ri, de_ti, layer_info_list, T1, kx_vector = self._solve(wavelength, e_conv_all, o_e_conv_all)

        self.layer_info_list = layer_info_list
        self.T1 = T1
        self.kx_vector = kx_vector

        return de_ri, de_ti

    def conv_solve(self, **kwargs):
        [setattr(self, k, v) for k, v in kwargs.items()]  # needed for optimization

        if self.fft_type == 0:
            E_conv_all, o_E_conv_all = to_conv_mat_discrete(self.ucell, self.fourier_order[0], self.fourier_order[1],
                                                            device=self.device, type_complex=self.type_complex,
                                                            improve_dft=self.improve_dft)
        elif self.fft_type == 1:
            E_conv_all, o_E_conv_all = to_conv_mat_continuous(self.ucell, self.fourier_order[0], self.fourier_order[1],
                                                              device=self.device, type_complex=self.type_complex)
        elif self.fft_type == 2:
            E_conv_all, o_E_conv_all = to_conv_mat_continuous_vector(self.ucell_info_list, self.fourier_order[0],
                                                                     self.fourier_order[1],
                                                                     type_complex=self.type_complex)
        else:
            raise ValueError

        de_ri, de_ti, layer_info_list, T1, kx_vector = self._solve(self.wavelength, E_conv_all, o_E_conv_all)

        self.layer_info_list = layer_info_list
        self.T1 = T1
        self.kx_vector = kx_vector

        return de_ri, de_ti

    def calculate_field(self, res_x=20, res_y=20, res_z=20, field_algo=2):
        if self.grating_type == 0:
            res_y = 1
            if field_algo == 0:
                field_cell = field_dist_1d_vanilla(self.wavelength, self.kx_vector,
                                                   self.T1, self.layer_info_list, self.period, self.pol,
                                                   res_x=res_x, res_y=res_y, res_z=res_z,
                                                   device=self.device, type_complex=self.type_complex)
            elif field_algo == 1:
                field_cell = field_dist_1d_vectorized_ji(self.wavelength, self.kx_vector, self.T1, self.layer_info_list,
                                                         self.period, self.pol, res_x=res_x, res_y=res_y, res_z=res_z,
                                                         device=self.device, type_complex=self.type_complex,
                                                         type_float=self.type_float)
            elif field_algo == 2:
                field_cell = field_dist_1d_vectorized_kji(self.wavelength, self.kx_vector, self.T1,
                                                          self.layer_info_list, self.period, self.pol,
                                                          res_x=res_x, res_y=res_y, res_z=res_z,
                                                          device=self.device, type_complex=self.type_complex,
                                                          type_float=self.type_float)
            else:
                raise ValueError
        elif self.grating_type == 1:
            res_y = 1
            if field_algo == 0:
                field_cell = field_dist_1d_conical_vanilla(self.wavelength, self.kx_vector, self.n_I, self.theta,
                                                           self.phi, self.T1, self.layer_info_list, self.period,
                                                           res_x=res_x, res_y=res_y, res_z=res_z, device=self.device,
                                                           type_complex=self.type_complex)
            elif field_algo == 1:
                field_cell = field_dist_1d_conical_vectorized_ji(self.wavelength, self.kx_vector, self.n_I, self.theta,
                                                                 self.phi, self.T1, self.layer_info_list, self.period,
                                                                 res_x=res_x, res_y=res_y, res_z=res_z,
                                                                 device=self.device,
                                                                 type_complex=self.type_complex,
                                                                 type_float=self.type_float)
            elif field_algo == 2:
                field_cell = field_dist_1d_conical_vectorized_kji(self.wavelength, self.kx_vector, self.n_I, self.theta,
                                                                  self.phi, self.T1, self.layer_info_list, self.period,
                                                                  res_x=res_x, res_y=res_y, res_z=res_z,
                                                                  device=self.device,
                                                                  type_complex=self.type_complex,
                                                                  type_float=self.type_float)
            else:
                raise ValueError

        elif self.grating_type == 2:
            if field_algo == 0:
                field_cell = field_dist_2d_vanilla(self.wavelength, self.kx_vector, self.n_I, self.theta, self.phi,
                                                   self.fourier_order[0], self.fourier_order[1], self.T1,
                                                   self.layer_info_list, self.period,
                                                   res_x=res_x, res_y=res_y, res_z=res_z, device=self.device,
                                                   type_complex=self.type_complex, type_float=self.type_float)
            elif field_algo == 1:
                field_cell = field_dist_2d_vectorized_ji(self.wavelength, self.kx_vector, self.n_I, self.theta,
                                                         self.phi, self.fourier_order[0], self.fourier_order[1],
                                                         self.T1, self.layer_info_list,
                                                         self.period, res_x=res_x, res_y=res_y, res_z=res_z,
                                                         device=self.device,
                                                         type_complex=self.type_complex, type_float=self.type_float)
            elif field_algo == 2:
                field_cell = field_dist_2d_vectorized_kji(self.wavelength, self.kx_vector, self.n_I, self.theta,
                                                          self.phi, self.fourier_order[0], self.fourier_order[1],
                                                          self.T1, self.layer_info_list,
                                                          self.period, res_x=res_x, res_y=res_y, res_z=res_z,
                                                          device=self.device,
                                                          type_complex=self.type_complex, type_float=self.type_float)
            else:
                raise ValueError
        else:
            raise ValueError

        return field_cell

    def conv_solve_field(self, res_x=20, res_y=20, res_z=20, field_algo=2):
        de_ri, de_ti = self.conv_solve()
        field_cell = self.calculate_field(res_x, res_y, res_z, field_algo=field_algo)
        return de_ri, de_ti, field_cell

    def field_plot(self, field_cell):
        field_plot(field_cell, self.pol)

    def calculate_field_all(self, res_x=20, res_y=20, res_z=20):
        t0 = time.time()
        field_cell0 = self.calculate_field(res_x=res_x, res_y=res_y, res_z=res_z, field_algo=0)
        print('no vector', time.time() - t0)
        t0 = time.time()
        field_cell1 = self.calculate_field(res_x=res_x, res_y=res_y, res_z=res_z, field_algo=1)
        print('ji vector', time.time() - t0)
        t0 = time.time()
        field_cell2 = self.calculate_field(res_x=res_x, res_y=res_y, res_z=res_z, field_algo=2)
        print('kji vector', time.time() - t0)

        print('gap(1-0): ', torch.linalg.norm(field_cell1 - field_cell0))
        print('gap(2-1): ', torch.linalg.norm(field_cell2 - field_cell1))
        print('gap(0-2): ', torch.linalg.norm(field_cell0 - field_cell2))

        return field_cell0, field_cell1, field_cell2
