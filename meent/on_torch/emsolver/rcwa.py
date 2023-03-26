import time
import torch

from ._base import _BaseRCWA
from .convolution_matrix import to_conv_mat_discrete, to_conv_mat_continuous, to_conv_mat_continuous_vector
from .field_distribution import field_dist_1d_vanilla, field_dist_2d_vanilla, field_plot, field_dist_1d_conical_vanilla, \
    field_dist_1d_vectorized_ji, field_dist_1d_vectorized_kji, field_dist_1d_conical_vectorized_ji, \
    field_dist_1d_conical_vectorized_kji, field_dist_2d_vectorized_ji, field_dist_2d_vectorized_kji


class RCWATorch(_BaseRCWA):
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
                 backend=2,
                 grating_type=0,
                 pol=0,
                 fourier_order=2,
                 ucell_materials=None,
                 algo='TMM',
                 perturbation=1E-10,
                 device='cpu',
                 type_complex=torch.complex128,
                 fft_type=0,
                 improve_dft=True,
                 **kwargs,
                 ):

        super().__init__(grating_type=grating_type, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi, pol=pol,
                         fourier_order=fourier_order, period=period, wavelength=wavelength,
                         thickness=thickness, algo=algo, perturbation=perturbation,
                         device=device, type_complex=type_complex)

        self.ucell = ucell.clone()
        self.ucell_materials = ucell_materials
        self.ucell_info_list = ucell_info_list

        self.backend = backend
        self.device = device
        self.type_complex = type_complex
        self.fft_type = fft_type
        self.improve_dft = improve_dft

        self.layer_info_list = []

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

        return de_ri.real, de_ti.real, layer_info_list, T1, self.kx_vector

    def solve(self, wavelength, e_conv_all, o_e_conv_all):
        de_ri, de_ti, layer_info_list, T1, kx_vector = self._solve(wavelength, e_conv_all, o_e_conv_all)

        self.layer_info_list = layer_info_list
        self.T1 = T1
        self.kx_vector = kx_vector

        return de_ri, de_ti

    def conv_solve(self, **kwargs):
        [setattr(self, k, v) for k, v in kwargs.items()]  # TODO: need this? for optimization?

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

    def calculate_field(self, resolution=None, plot=True, field_algo=2):

        if self.grating_type == 0:
            resolution = [100, 1, 100] if not resolution else resolution

            if field_algo == 0:
                field_cell = field_dist_1d_vanilla(self.wavelength, self.kx_vector,
                                                   self.T1, self.layer_info_list, self.period, self.pol,
                                                   resolution=resolution, type_complex=self.type_complex)
            elif field_algo == 1:
                field_cell = field_dist_1d_vectorized_ji(self.wavelength, self.kx_vector, self.T1, self.layer_info_list,
                                                         self.period, self.pol, resolution=resolution,
                                                         type_complex=self.type_complex)
            elif field_algo == 2:
                field_cell = field_dist_1d_vectorized_kji(self.wavelength, self.kx_vector, self.T1,
                                                          self.layer_info_list, self.period, self.pol,
                                                          resolution=resolution, type_complex=self.type_complex)
            else:
                raise ValueError

        elif self.grating_type == 1:
            resolution = [100, 1, 100] if not resolution else resolution

            if field_algo == 0:
                field_cell = field_dist_1d_conical_vanilla(self.wavelength, self.kx_vector, self.n_I, self.theta,
                                                           self.phi, self.T1, self.layer_info_list, self.period,
                                                           resolution=resolution, device=self.device, type_complex=self.type_complex)
            elif field_algo == 1:
                field_cell = field_dist_1d_conical_vectorized_ji(self.wavelength, self.kx_vector, self.n_I, self.theta,
                                                                 self.phi, self.T1, self.layer_info_list, self.period,
                                                                 resolution=resolution, device=self.device, type_complex=self.type_complex)
            elif field_algo == 2:
                field_cell = field_dist_1d_conical_vectorized_kji(self.wavelength, self.kx_vector, self.n_I, self.theta,
                                                                  self.phi, self.T1, self.layer_info_list, self.period,
                                                                  resolution=resolution, device=self.device, type_complex=self.type_complex)
            else:
                raise ValueError

        elif self.grating_type == 2:
            resolution = [10, 10, 10] if not resolution else resolution

            if field_algo == 0:
                field_cell = field_dist_2d_vanilla(self.wavelength, self.kx_vector, self.n_I, self.theta, self.phi,
                                                   *self.fourier_order, self.T1, self.layer_info_list, self.period,
                                                   resolution=resolution, device=self.device, type_complex=self.type_complex)
            elif field_algo == 1:
                field_cell = field_dist_2d_vectorized_ji(self.wavelength, self.kx_vector, self.n_I, self.theta,
                                                         self.phi, *self.fourier_order, self.T1, self.layer_info_list,
                                                         self.period, resolution=resolution, device=self.device,
                                                         type_complex=self.type_complex)
            elif field_algo == 2:
                field_cell = field_dist_2d_vectorized_kji(self.wavelength, self.kx_vector, self.n_I, self.theta,
                                                          self.phi, *self.fourier_order, self.T1, self.layer_info_list,
                                                          self.period, resolution=resolution, device=self.device,
                                                          type_complex=self.type_complex)
            else:
                raise ValueError
        else:
            raise ValueError

        if plot:
            field_plot(field_cell, self.pol)

        return field_cell

    def calculate_field_all(self, resolution=None, plot=True):

        if self.grating_type == 0:
            resolution = [100, 1, 100] if not resolution else resolution

            t0 = time.time()
            field_cell0 = field_dist_1d_vanilla(self.wavelength, self.kx_vector,
                                                self.T1, self.layer_info_list, self.period, self.pol,
                                                resolution=resolution,
                                                type_complex=self.type_complex)
            print('no vector', time.time() - t0)

            t0 = time.time()
            field_cell1 = field_dist_1d_vectorized_ji(self.wavelength, self.kx_vector,
                                                      self.T1, self.layer_info_list, self.period, self.pol,
                                                      resolution=resolution,
                                                      type_complex=self.type_complex)
            print('ji vector', time.time() - t0)

            t0 = time.time()
            field_cell2 = field_dist_1d_vectorized_kji(self.wavelength, self.kx_vector,
                                                       self.T1, self.layer_info_list, self.period, self.pol,
                                                       resolution=resolution,
                                                       type_complex=self.type_complex)
            print('kji vector', time.time() - t0)

            print('gap: ', torch.linalg.norm(field_cell1 - field_cell0))
            print('gap: ', torch.linalg.norm(field_cell2 - field_cell0))

        elif self.grating_type == 1:
            resolution = [100, 1, 100] if not resolution else resolution

            t0 = time.time()
            field_cell0 = field_dist_1d_conical_vanilla(self.wavelength, self.kx_vector, self.n_I, self.theta, self.phi,
                                                        self.T1, self.layer_info_list, self.period,
                                                        resolution=resolution, device=self.device,
                                                        type_complex=self.type_complex)
            print('no vector', time.time() - t0)

            t0 = time.time()
            field_cell1 = field_dist_1d_conical_vectorized_ji(self.wavelength, self.kx_vector, self.n_I, self.theta,
                                                              self.phi,
                                                              self.T1, self.layer_info_list, self.period,
                                                              resolution=resolution, device=self.device,
                                                              type_complex=self.type_complex)
            print('ji vector', time.time() - t0)

            t0 = time.time()
            field_cell2 = field_dist_1d_conical_vectorized_kji(self.wavelength, self.kx_vector, self.n_I, self.theta,
                                                               self.phi,
                                                               self.T1, self.layer_info_list, self.period,
                                                               resolution=resolution, device=self.device,
                                                               type_complex=self.type_complex)
            print('kji vector', time.time() - t0)

            print('gap: ', torch.linalg.norm(field_cell1 - field_cell0))
            print('gap: ', torch.linalg.norm(field_cell2 - field_cell0))

        else:
            resolution = [10, 10, 10] if not resolution else resolution

            t0 = time.time()
            field_cell0 = field_dist_2d_vanilla(self.wavelength, self.kx_vector, self.n_I, self.theta,
                                                self.phi, *self.fourier_order,
                                                self.T1, self.layer_info_list, self.period,
                                                resolution=resolution, device=self.device,
                                                type_complex=self.type_complex)
            print('no vector', time.time() - t0)

            t0 = time.time()
            field_cell1 = field_dist_2d_vectorized_ji(self.wavelength, self.kx_vector, self.n_I, self.theta, self.phi,
                                                      *self.fourier_order,
                                                      self.T1, self.layer_info_list, self.period, resolution=resolution,
                                                      device=self.device,
                                                      type_complex=self.type_complex)
            print('ji vector', time.time() - t0)

            t0 = time.time()
            field_cell2 = field_dist_2d_vectorized_kji(self.wavelength, self.kx_vector, self.n_I, self.theta,
                                                       self.phi, *self.fourier_order,
                                                       self.T1, self.layer_info_list, self.period,
                                                       resolution=resolution, device=self.device,
                                                       type_complex=self.type_complex)
            print('kji vector', time.time() - t0)

            print('gap: ', torch.linalg.norm(field_cell1 - field_cell0))
            print('gap: ', torch.linalg.norm(field_cell2 - field_cell0))

        if plot:
            field_plot(field_cell0, self.pol)
            field_plot(field_cell1, self.pol)
            field_plot(field_cell2, self.pol)

        return

    # def calculate_field(self, resolution=None, plot=True):
    #
    #     if self.grating_type == 0:
    #         resolution = [100, 1, 100] if not resolution else resolution
    #         # field_cell = field_dist_1d_vanilla(self.wavelength, self.kx_vector,
    #         #                                    self.T1, self.layer_info_list, self.period, self.pol, resolution=resolution,
    #         #                                    device=self.device, type_complex=self.type_complex)
    #         t0 = time.time()
    #         field_cell1 = field_dist_1d_vanilla(self.wavelength, self.kx_vector,
    #                                             self.T1, self.layer_info_list, self.period, self.pol,
    #                                             resolution=resolution,
    #                                             type_complex=self.type_complex)
    #         print('no vector', time.time() - t0)
    #         # field_plot(field_cell1, self.pol)
    #
    #         t0 = time.time()
    #         field_cell = field_dist_1d_vectorized_ji(self.wavelength, self.kx_vector,
    #                                                  self.T1, self.layer_info_list, self.period, self.pol,
    #                                                  resolution=resolution,
    #                                                  type_complex=self.type_complex)
    #         print('ji vector', time.time() - t0)
    #         # field_plot(field_cell, self.pol)
    #
    #         t0 = time.time()
    #         field_cell2 = field_dist_1d_vectorized_kji(self.wavelength, self.kx_vector,
    #                                                    self.T1, self.layer_info_list, self.period, self.pol,
    #                                                    resolution=resolution,
    #                                                    type_complex=self.type_complex)
    #         # field_plot(field_cell2, self.pol)
    #
    #         print('kji vector', time.time() - t0)
    #         print('gap: ', torch.linalg.norm(field_cell - field_cell1))
    #         print('gap: ', torch.linalg.norm(field_cell2 - field_cell1))
    #
    #     elif self.grating_type == 1:
    #         resolution = [100, 1, 100] if not resolution else resolution
    #         # field_cell = field_dist_1d_conical_vanilla(self.wavelength, self.kx_vector, self.n_I, self.theta, self.phi,
    #         #                                            self.T1, self.layer_info_list, self.period,
    #         #                                            resolution=resolution, device=self.device,
    #         #                                            type_complex=self.type_complex)
    #         t0 = time.time()
    #         field_cell1 = field_dist_1d_conical_vanilla(self.wavelength, self.kx_vector, self.n_I, self.theta, self.phi,
    #                                                     self.T1, self.layer_info_list, self.period,
    #                                                     resolution=resolution,
    #                                                     type_complex=self.type_complex)
    #         print('no vector', time.time() - t0)
    #
    #         t0 = time.time()
    #         field_cell = field_dist_1d_conical_vectorized_ji(self.wavelength, self.kx_vector, self.n_I, self.theta,
    #                                                          self.phi,
    #                                                          self.T1, self.layer_info_list, self.period,
    #                                                          resolution=resolution,
    #                                                          type_complex=self.type_complex)
    #         print('ji vector', time.time() - t0)
    #
    #         t0 = time.time()
    #         field_cell2 = field_dist_1d_conical_vectorized_kji(self.wavelength, self.kx_vector, self.n_I, self.theta,
    #                                                            self.phi,
    #                                                            self.T1, self.layer_info_list, self.period,
    #                                                            resolution=resolution,
    #                                                            type_complex=self.type_complex)
    #         print('kji vector', time.time() - t0)
    #         print('gap: ', torch.linalg.norm(field_cell - field_cell1))
    #         print('gap: ', torch.linalg.norm(field_cell2 - field_cell1))
    #
    #     else:
    #         resolution = [10, 10, 10] if not resolution else resolution
    #         # field_cell = field_dist_2d_vanilla(self.wavelength, self.kx_vector, self.n_I, self.theta, self.phi,
    #         #                                    *self.fourier_order, self.T1, self.layer_info_list, self.period,
    #         #                                    resolution=resolution, device=self.device, type_complex=self.type_complex)
    #
    #         # t0 = time.time()
    #         # field_cell1 = field_dist_2d_vanilla(self.wavelength, self.kx_vector, self.n_I, self.theta,
    #         #                                     self.phi, *self.fourier_order,
    #         #                                     self.T1, self.layer_info_list, self.period,
    #         #                                     resolution=resolution,
    #         #                                     type_complex=self.type_complex)
    #         # print('no vector', time.time() - t0)
    #
    #         # t0 = time.time()
    #         # field_cell = field_dist_2d_vectorized_ji(self.wavelength, self.kx_vector, self.n_I, self.theta, self.phi,
    #         #                                          *self.fourier_order,
    #         #                                          self.T1, self.layer_info_list, self.period, resolution=resolution,
    #         #                                          type_complex=self.type_complex)
    #         # print('ji vector', time.time() - t0)
    #
    #         t0 = time.time()
    #         field_cell = field_dist_2d_vectorized_kji(self.wavelength, self.kx_vector, self.n_I, self.theta,
    #                                                    self.phi, *self.fourier_order,
    #                                                    self.T1, self.layer_info_list, self.period,
    #                                                    resolution=resolution,
    #                                                   device=self.device,
    #                                                    type_complex=self.type_complex)
    #         print('kji vector', time.time() - t0)
    #         # print('gap: ', torch.linalg.norm(field_cell - field_cell1))
    #         # print('gap: ', torch.linalg.norm(field_cell2 - field_cell1))
    #
    #     if plot:
    #         field_plot(field_cell, self.pol)
    #
    #     return field_cell
