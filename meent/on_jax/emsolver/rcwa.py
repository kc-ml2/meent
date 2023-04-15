import time
from functools import partial

import jax

import numpy as np
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
                 theta=0.,
                 phi=0.,
                 period=(100., 100.),
                 wavelength=900.,
                 ucell=None,
                 ucell_info_list=None,
                 thickness=(0., ),
                 backend=1,
                 grating_type=0,
                 pol=0.,
                 fourier_order=(2, 0),
                 ucell_materials=None,
                 algo='TMM',
                 perturbation=1E-20,
                 device='cpu',
                 type_complex=jnp.complex128,
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

    @property
    def ucell(self):
        return self._ucell

    @ucell.setter
    def ucell(self, ucell):
        if isinstance(ucell, jnp.ndarray):
            if ucell.dtype in (jnp.float64, jnp.float32, jnp.int64, jnp.int32):
                dtype = self.type_float
            elif ucell.dtype in (jnp.complex128, jnp.complex64):
                dtype = self.type_complex
            else:
                raise ValueError
            self._ucell = ucell.astype(dtype)
        elif isinstance(ucell, np.ndarray):
            if ucell.dtype in (np.int64, np.float64, np.int32, np.float32):
                dtype = self.type_float
            elif ucell.dtype in (np.complex128, np.complex64):
                dtype = self.type_complex
            else:
                raise ValueError
            self._ucell = jnp.array(ucell, dtype=dtype)
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

    @_BaseRCWA.jax_device_set
    def solve(self, wavelength, e_conv_all, o_e_conv_all):
        de_ri, de_ti, layer_info_list, T1, kx_vector = jax.jit(self._solve)(wavelength, e_conv_all, o_e_conv_all)

        self.layer_info_list = layer_info_list
        self.T1 = T1
        self.kx_vector = kx_vector

        return de_ri, de_ti

    def _conv_solve(self):

        if self.fft_type == 0:
            E_conv_all, o_E_conv_all = to_conv_mat_discrete(self.ucell, self.fourier_order[0], self.fourier_order[1],
                                              type_complex=self.type_complex, improve_dft=self.improve_dft)
        elif self.fft_type == 1:
            E_conv_all, o_E_conv_all = to_conv_mat_continuous(self.ucell, self.fourier_order[0], self.fourier_order[1],
                                                type_complex=self.type_complex)
        elif self.fft_type == 2:
            E_conv_all, o_E_conv_all = to_conv_mat_continuous_vector(self.ucell_info_list,
                                                                     self.fourier_order[0], self.fourier_order[1],
                                                                     type_complex=self.type_complex)
        else:
            raise ValueError

        de_ri, de_ti, layer_info_list, T1, kx_vector = self._solve(self.wavelength, E_conv_all, o_E_conv_all)
        return de_ri, de_ti, layer_info_list, T1, kx_vector

    @jax.jit
    def _conv_solve_jit(self):
        return self._conv_solve()

    @_BaseRCWA.jax_device_set
    def conv_solve(self, **kwargs):
        [setattr(self, k, v) for k, v in kwargs.items()]  # needed for optimization
        if self.fft_type == 1:
            # print('CFT (fft_type=1) is not supported for jit-compilation. Using non-jit-compiled method.')
            de_ri, de_ti, layer_info_list, T1, kx_vector = self._conv_solve()

        else:
            de_ri, de_ti, layer_info_list, T1, kx_vector = self._conv_solve_jit()

        self.layer_info_list = layer_info_list
        self.T1 = T1
        self.kx_vector = kx_vector

        return de_ri, de_ti

    @_BaseRCWA.jax_device_set
    def calculate_field(self, res_x=20, res_y=20, res_z=20, field_algo=2):

        if self.grating_type == 0:
            res_y = 1
            if field_algo == 0:
                field_cell = field_dist_1d_vanilla(self.wavelength, self.kx_vector,
                                                   self.T1, self.layer_info_list, self.period, self.pol,
                                                   res_x=res_x, res_y=res_y, res_z=res_z,
                                                   type_complex=self.type_complex)
            elif field_algo == 1:
                field_cell = field_dist_1d_vectorized_ji(self.wavelength, self.kx_vector, self.T1, self.layer_info_list,
                                                         self.period, self.pol, res_x=res_x, res_y=res_y, res_z=res_z,
                                                         type_complex=self.type_complex, type_float=self.type_float)
            elif field_algo == 2:
                field_cell = field_dist_1d_vectorized_kji(self.wavelength, self.kx_vector, self.T1,
                                                          self.layer_info_list, self.period, self.pol,
                                                          res_x=res_x, res_y=res_y, res_z=res_z,
                                                          type_complex=self.type_complex, type_float=self.type_float)
            else:
                raise ValueError

        elif self.grating_type == 1:
            res_y = 1
            if field_algo == 0:
                field_cell = field_dist_1d_conical_vanilla(self.wavelength, self.kx_vector, self.n_I, self.theta,
                                                           self.phi, self.T1, self.layer_info_list, self.period,
                                                           res_x=res_x, res_y=res_y, res_z=res_z,
                                                           type_complex=self.type_complex)
            elif field_algo == 1:
                field_cell = field_dist_1d_conical_vectorized_ji(self.wavelength, self.kx_vector, self.n_I, self.theta,
                                                                 self.phi, self.T1, self.layer_info_list, self.period,
                                                                 res_x=res_x, res_y=res_y, res_z=res_z,
                                                                 type_complex=self.type_complex, type_float=self.type_float)
            elif field_algo == 2:
                field_cell = field_dist_1d_conical_vectorized_kji(self.wavelength, self.kx_vector, self.n_I, self.theta,
                                                                  self.phi, self.T1, self.layer_info_list, self.period,
                                                                  res_x=res_x, res_y=res_y, res_z=res_z,
                                                                  type_complex=self.type_complex, type_float=self.type_float)
            else:
                raise ValueError

        elif self.grating_type == 2:

            if field_algo == 0:
                field_cell = field_dist_2d_vanilla(self.wavelength, self.kx_vector, self.n_I, self.theta, self.phi,
                                                   *self.fourier_order, self.T1, self.layer_info_list, self.period,
                                                   res_x=res_x, res_y=res_y, res_z=res_z,
                                                   type_complex=self.type_complex)
            elif field_algo == 1:
                field_cell = field_dist_2d_vectorized_ji(self.wavelength, self.kx_vector, self.n_I, self.theta,
                                                         self.phi, *self.fourier_order, self.T1, self.layer_info_list,
                                                         self.period, res_x=res_x, res_y=res_y, res_z=res_z,
                                                         type_complex=self.type_complex, type_float=self.type_float)
            elif field_algo == 2:
                field_cell = field_dist_2d_vectorized_kji(self.wavelength, self.kx_vector, self.n_I, self.theta,
                                                          self.phi, *self.fourier_order, self.T1, self.layer_info_list,
                                                          self.period, res_x=res_x, res_y=res_y, res_z=res_z,
                                                          type_complex=self.type_complex, type_float=self.type_float)
            else:
                raise ValueError
        else:
            raise ValueError

        return field_cell

    def field_plot(self, field_cell):
        field_plot(field_cell, self.pol)

    @_BaseRCWA.jax_device_set
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

        print('gap(1-0): ', jnp.linalg.norm(field_cell1 - field_cell0))
        print('gap(2-1): ', jnp.linalg.norm(field_cell2 - field_cell1))
        print('gap(0-2): ', jnp.linalg.norm(field_cell0 - field_cell2))

        return field_cell0, field_cell1, field_cell2

    @partial(jax.jit, static_argnums=(1, 2, 3, 4))
    @_BaseRCWA.jax_device_set
    def conv_solve_field(self, res_x=20, res_y=20, res_z=20, field_algo=2):

        if self.fft_type == 1:
            print('CFT (fft_type=1) is not supported with JAX jit-compilation. Use conv_solve_field_no_jit.')
            return None, None, None

        de_ri, de_ti, _, _, _ = self._conv_solve()
        field_cell = self.calculate_field(res_x, res_y, res_z, field_algo=field_algo)
        return de_ri, de_ti, field_cell

    @_BaseRCWA.jax_device_set
    def conv_solve_field_no_jit(self, res_x=20, res_y=20, res_z=20, field_algo=2):
        de_ri, de_ti, _, _, _ = self._conv_solve()
        field_cell = self.calculate_field(res_x, res_y, res_z, field_algo=field_algo)
        return de_ri, de_ti, field_cell

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
