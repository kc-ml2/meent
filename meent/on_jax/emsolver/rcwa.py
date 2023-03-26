import time
from copy import deepcopy
from functools import partial

import jax
import jax.numpy as jnp

from ._base import _BaseRCWA
from .convolution_matrix import to_conv_mat_discrete, to_conv_mat_continuous, to_conv_mat_continuous_vector
from .field_distribution import field_dist_1d_vectorized_ji, field_dist_1d_conical_vectorized_ji, \
    field_dist_2d_vectorized_ji, field_plot, \
    field_dist_1d_vectorized_kji, field_dist_1d_conical_vectorized_kji, field_dist_1d_vanilla, \
    field_dist_1d_conical_vanilla, field_dist_2d_vanilla, field_dist_2d_vectorized_kji


class RCWAJax(_BaseRCWA):
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
                 backend=1,
                 grating_type=0,
                 pol=0,
                 fourier_order=2,
                 ucell_materials=None,
                 algo='TMM',
                 perturbation=1E-10,
                 device='cpu',
                 type_complex=jnp.complex128,
                 fft_type=0,
                 improve_dft=True,
                 **kwargs,
                 ):

        super().__init__(grating_type=grating_type, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi, pol=pol,
                         fourier_order=fourier_order, period=period, wavelength=wavelength,
                         thickness=thickness, algo=algo, perturbation=perturbation,
                         device=device, type_complex=type_complex)

        self.ucell = deepcopy(ucell)
        self.ucell_materials = ucell_materials
        self.ucell_info_list = ucell_info_list

        self.backend = backend
        self.device = device
        self.type_complex = type_complex
        self.fft_type = fft_type
        self.improve_dft = improve_dft

        self.layer_info_list = []

    def _tree_flatten(self):
        children = (self.n_I, self.n_II, self.theta, self.phi, self.psi,
                    self.period, self.wavelength, self.ucell, self.ucell_info_list, self.thickness)
        aux_data = {
            'backend': self.backend,
            'grating_type': self.grating_type,
            'pol': self.pol,
            'fourier_order': self.fourier_order,
            'ucell_materials': self.ucell_materials,
            'algo': self.algo,
            'perturbation': self.perturbation,
            'device': self.device,
            'type_complex': self.type_complex,
            'fft_type': self.fft_type,
        }

        return children, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    # @jax.jit
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
        de_ri, de_ti, layer_info_list, T1, kx_vector = jax.jit(self._solve)(wavelength, e_conv_all, o_e_conv_all)

        self.layer_info_list = layer_info_list
        self.T1 = T1
        self.kx_vector = kx_vector

        return de_ri, de_ti

    # @jax.jit  # TODO: can draw field? with jit?
    def _conv_solve(self):

        if self.fft_type == 0:
            # if self.ucell.shape[1] == 1:
            #     ff = 2 * self.fourier_order[0] + 1
            #
            #     if self.improve_dft:
            #         minimum_pattern_size_x = 2 * ff * self.ucell.shape[2]
            #     else:
            #         minimum_pattern_size_x = 2 * ff
            #     n_y = 0
            #     n_x = minimum_pattern_size_x // self.ucell.shape[2]
            #
            # else:
            #     ff_x = 2 * self.fourier_order[0] + 1
            #     ff_y = 2 * self.fourier_order[1] + 1
            #
            #     if self.improve_dft:
            #         minimum_pattern_size_y = 2 * ff_y * self.ucell.shape[1]
            #         minimum_pattern_size_x = 2 * ff_x * self.ucell.shape[2]
            #     else:
            #         minimum_pattern_size_y = 2 * ff_y
            #         minimum_pattern_size_x = 2 * ff_x
            #
            #     if self.ucell.shape[1] < minimum_pattern_size_y:
            #         n_y = minimum_pattern_size_y // self.ucell.shape[1]
            #     else:
            #         n_y = 0
            #     if self.ucell.shape[2] < minimum_pattern_size_x:
            #         n_x = minimum_pattern_size_x // self.ucell.shape[2]
            #     else:
            #         n_x = 0

            E_conv_all = to_conv_mat_discrete(self.ucell, self.fourier_order[0], self.fourier_order[1],
                                              type_complex=self.type_complex, improve_dft=self.improve_dft)
            o_E_conv_all = to_conv_mat_discrete(1 / self.ucell, self.fourier_order[0], self.fourier_order[1],
                                                type_complex=self.type_complex, improve_dft=self.improve_dft)
        elif self.fft_type == 2:
            E_conv_all, o_E_conv_all = to_conv_mat_continuous_vector(self.ucell_info_list, self.fourier_order,
                                                                     type_complex=self.type_complex)
        else:
            raise ValueError

        de_ri, de_ti, layer_info_list, T1, kx_vector = self._solve(self.wavelength, E_conv_all, o_E_conv_all)
        # self.layer_info_list = layer_info_list
        # self.T1 = T1
        # self.kx_vector = kx_vector
        return de_ri, de_ti, layer_info_list, T1, kx_vector

    def _conv_solve_no_jit(self):

        E_conv_all = to_conv_mat_continuous(self.ucell, self.fourier_order[0], self.fourier_order[1],
                                            type_complex=self.type_complex)
        o_E_conv_all = to_conv_mat_continuous(1 / self.ucell, self.fourier_order[0], self.fourier_order[1],
                                              type_complex=self.type_complex)

        de_ri, de_ti, layer_info_list, T1, kx_vector = self._solve(self.wavelength, E_conv_all, o_E_conv_all)

        return de_ri, de_ti, layer_info_list, T1, kx_vector

    def conv_solve(self, **kwargs):
        [setattr(self, k, v) for k, v in kwargs.items()]  # TODO: need this? for optimization?

        if self.fft_type == 1:
            de_ri, de_ti, layer_info_list, T1, kx_vector = self._conv_solve_no_jit()
        else:
            de_ri, de_ti, layer_info_list, T1, kx_vector = self._conv_solve()

        self.layer_info_list = layer_info_list
        self.T1 = T1
        self.kx_vector = kx_vector

        return de_ri, de_ti

    @jax.jit
    def conv_solve_spectrum(self, ucell):  # TODO: other backends
        E_conv_all = to_conv_mat_discrete(ucell, self.fourier_order[0], self.fourier_order[1],
                                          type_complex=self.type_complex, improve_dft=self.improve_dft)
        o_E_conv_all = to_conv_mat_discrete(1 / ucell, self.fourier_order[0], self.fourier_order[1],
                                            type_complex=self.type_complex, improve_dft=self.improve_dft)
        de_ri, de_ti, layer_info_list, T1, kx_vector = self._solve(self.wavelength, E_conv_all, o_E_conv_all)
        return de_ri, de_ti

    def run_ucell_vmap(self, ucell_list):
        """
        under development
        """

        de_ri, de_ti = jax.vmap(self.conv_solve)(ucell_list)

        return de_ri, de_ti

    def run_ucell_pmap(self, ucell_list):
        """
        under development
        """

        de_ri, de_ti = jax.pmap(self.conv_solve)(ucell_list)
        # de_ri.block_until_ready()
        # de_ti.block_until_ready()

        de_ri = jnp.array(de_ri)
        de_ti = jnp.array(de_ti)
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
                                                           resolution=resolution, type_complex=self.type_complex)
            elif field_algo == 1:
                field_cell = field_dist_1d_conical_vectorized_ji(self.wavelength, self.kx_vector, self.n_I, self.theta,
                                                                 self.phi, self.T1, self.layer_info_list, self.period,
                                                                 resolution=resolution, type_complex=self.type_complex)
            elif field_algo == 2:
                field_cell = field_dist_1d_conical_vectorized_kji(self.wavelength, self.kx_vector, self.n_I, self.theta,
                                                                  self.phi, self.T1, self.layer_info_list, self.period,
                                                                  resolution=resolution, type_complex=self.type_complex)
            else:
                raise ValueError

        elif self.grating_type == 2:
            resolution = [10, 10, 10] if not resolution else resolution

            if field_algo == 0:
                field_cell = field_dist_2d_vanilla(self.wavelength, self.kx_vector, self.n_I, self.theta, self.phi,
                                                   *self.fourier_order, self.T1, self.layer_info_list, self.period,
                                                   resolution=resolution, type_complex=self.type_complex)
            elif field_algo == 1:
                field_cell = field_dist_2d_vectorized_ji(self.wavelength, self.kx_vector, self.n_I, self.theta,
                                                         self.phi, *self.fourier_order, self.T1, self.layer_info_list,
                                                         self.period, resolution=resolution,
                                                         type_complex=self.type_complex)
            elif field_algo == 2:
                field_cell = field_dist_2d_vectorized_kji(self.wavelength, self.kx_vector, self.n_I, self.theta,
                                                          self.phi, *self.fourier_order, self.T1, self.layer_info_list,
                                                          self.period, resolution=resolution,
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

            print('gap: ', jnp.linalg.norm(field_cell1 - field_cell0))
            print('gap: ', jnp.linalg.norm(field_cell2 - field_cell0))

        elif self.grating_type == 1:
            resolution = [100, 1, 100] if not resolution else resolution

            t0 = time.time()
            field_cell0 = field_dist_1d_conical_vanilla(self.wavelength, self.kx_vector, self.n_I, self.theta, self.phi,
                                                        self.T1, self.layer_info_list, self.period,
                                                        resolution=resolution,
                                                        type_complex=self.type_complex)
            print('no vector', time.time() - t0)

            t0 = time.time()
            field_cell1 = field_dist_1d_conical_vectorized_ji(self.wavelength, self.kx_vector, self.n_I, self.theta,
                                                              self.phi,
                                                              self.T1, self.layer_info_list, self.period,
                                                              resolution=resolution,
                                                              type_complex=self.type_complex)
            print('ji vector', time.time() - t0)

            t0 = time.time()
            field_cell2 = field_dist_1d_conical_vectorized_kji(self.wavelength, self.kx_vector, self.n_I, self.theta,
                                                               self.phi,
                                                               self.T1, self.layer_info_list, self.period,
                                                               resolution=resolution,
                                                               type_complex=self.type_complex)
            print('kji vector', time.time() - t0)

            print('gap: ', jnp.linalg.norm(field_cell1 - field_cell0))
            print('gap: ', jnp.linalg.norm(field_cell2 - field_cell0))

        else:
            resolution = [10, 10, 10] if not resolution else resolution

            t0 = time.time()
            field_cell0 = field_dist_2d_vanilla(self.wavelength, self.kx_vector, self.n_I, self.theta,
                                                self.phi, *self.fourier_order,
                                                self.T1, self.layer_info_list, self.period,
                                                resolution=resolution,
                                                type_complex=self.type_complex)
            print('no vector', time.time() - t0)

            t0 = time.time()
            field_cell1 = field_dist_2d_vectorized_ji(self.wavelength, self.kx_vector, self.n_I, self.theta, self.phi,
                                                      *self.fourier_order,
                                                      self.T1, self.layer_info_list, self.period, resolution=resolution,
                                                      type_complex=self.type_complex)
            print('ji vector', time.time() - t0)

            t0 = time.time()
            field_cell2 = field_dist_2d_vectorized_kji(self.wavelength, self.kx_vector, self.n_I, self.theta,
                                                       self.phi, *self.fourier_order,
                                                       self.T1, self.layer_info_list, self.period,
                                                       resolution=resolution,
                                                       type_complex=self.type_complex)
            print('kji vector', time.time() - t0)

            print('gap: ', jnp.linalg.norm(field_cell1 - field_cell0))
            print('gap: ', jnp.linalg.norm(field_cell2 - field_cell0))

        if plot:
            field_plot(field_cell0, self.pol)
            field_plot(field_cell1, self.pol)
            field_plot(field_cell2, self.pol)

        return

    @jax.jit
    def conv_solve_calculate_field(self, resolution=None, plot=False):
        self._conv_solve()
        if self.grating_type == 0:

            resolution = [100, 1, 100] if not resolution else resolution
            field_cell = field_dist_1d_vectorized_ji(self.wavelength, self.kx_vector, self.n_I, self.theta,
                                                     self.fourier_order, self.T1,
                                                     self.layer_info_list, self.period, self.pol, resolution=resolution,
                                                     type_complex=self.type_complex)
        elif self.grating_type == 1:
            resolution = [100, 1, 100] if not resolution else resolution
            field_cell = field_dist_1d_conical_vectorized_ji(self.wavelength, self.kx_vector, self.n_I, self.theta,
                                                             self.phi,
                                                             self.fourier_order, self.T1, self.layer_info_list,
                                                             self.period,
                                                             resolution=resolution, type_complex=self.type_complex)

        else:
            resolution = [10, 10, 10] if not resolution else resolution
            field_cell = field_dist_2d_vectorized_ji(self.wavelength, self.kx_vector, self.n_I, self.theta, self.phi,
                                                     self.fourier_order, self.T1, self.layer_info_list, self.period,
                                                     resolution=resolution, type_complex=self.type_complex)
        if plot:
            field_plot(field_cell, self.pol)
        return field_cell

    # def generate_spectrum(self, wl_list):
    #     ucell = deepcopy(self.ucell)
    #     spectrum_ri_pmap = jnp.zeros(wl_list.shape)
    #     spectrum_ti_pmap = jnp.zeros(wl_list.shape)
    #     for i, wavelength in enumerate(range(counter)):
    #         b = i * num_device
    #         de_ri_pmap, de_ti_pmap = jax.pmap(loop_wavelength)(wavelength_list[b:b + num_device],
    #                                                            mat_pmtvy_interp[b:b + num_device])
    #
    #         spectrum_ri_pmap[b:b + num_device] = de_ri_pmap.sum(axis=(1, 2))
    #         spectrum_ti_pmap[b:b + num_device] = de_ti_pmap.sum(axis=(1, 2))
    #
    #     return spectrum_ri_pmap, spectrum_ti_pmap
