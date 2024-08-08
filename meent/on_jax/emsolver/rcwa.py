import jax

import numpy as np
import jax.numpy as jnp

from functools import partial

from ._base import _BaseRCWA, jax_device_set
from .convolution_matrix import to_conv_mat_raster_discrete, to_conv_mat_raster_continuous, to_conv_mat_vector
from .field_distribution import field_dist_1d, field_dist_2d,  field_plot


class RCWAJax(_BaseRCWA):
    def __init__(self,
                 n_top=1.,
                 n_bot=1.,
                 theta=0.,
                 phi=0.,
                 psi=None,
                 period=(100., 100.),
                 wavelength=900.,
                 ucell=None,
                 thickness=(0., ),
                 backend=0,
                 pol=0.,
                 fto=(0, 0),
                 ucell_materials=None,
                 connecting_algo='TMM',
                 perturbation=1E-20,
                 device='cpu',
                 type_complex=np.complex128,
                 fourier_type=0,  # 0 DFS, 1 CFS
                 enhanced_dfs=True,
                 # **kwargs,
                 ):

        super().__init__(n_top=n_top, n_bot=n_bot, theta=theta, phi=phi, psi=psi, pol=pol,
                         fto=fto, period=period, wavelength=wavelength,
                         thickness=thickness, connecting_algo=connecting_algo, perturbation=perturbation,
                         device=device, type_complex=type_complex)

        self.ucell = ucell
        self.ucell_materials = ucell_materials

        self.backend = backend
        self.fourier_type = fourier_type
        self.enhanced_dfs = enhanced_dfs
        self._modeling_type_assigned = None
        self._grating_type_assigned = None

    @property
    def ucell(self):
        return self._ucell

    @ucell.setter
    def ucell(self, ucell):

        if isinstance(ucell, jnp.ndarray):  # Raster
            if ucell.dtype in (jnp.float64, jnp.float32, jnp.int64, jnp.int32):
                dtype = self.type_float
                self._ucell = ucell.astype(dtype)
            elif ucell.dtype in (jnp.complex128, jnp.complex64):
                dtype = self.type_complex
                self._ucell = ucell.astype(dtype)

        elif isinstance(ucell, np.ndarray):  # Raster
            if ucell.dtype in (np.int64, np.float64, np.int32, np.float32):
                dtype = self.type_float
                self._ucell = jnp.array(ucell, dtype=dtype)
            elif ucell.dtype in (np.complex128, np.complex64):
                dtype = self.type_complex
                self._ucell = jnp.array(ucell, dtype=dtype)

        elif type(ucell) is list:  # Vector
            self._ucell = ucell
        elif ucell is None:
            self._ucell = ucell
        else:
            raise ValueError

    @property
    def modeling_type_assigned(self):
        return self._modeling_type_assigned

    @modeling_type_assigned.setter
    def modeling_type_assigned(self, modeling_type_assigned):
        self._modeling_type_assigned = modeling_type_assigned

    def _assign_modeling_type(self):
        if isinstance(self.ucell, (np.ndarray, jnp.ndarray)):  # Raster
            self.modeling_type_assigned = 0
            if (self.ucell.shape[1] == 1) and (self.pol in (0, 1)):

                def false_fun(): return 0  # 1D TE and TM only
                def true_fun(): return 1

                gear = jax.lax.cond(self.phi % (2 * np.pi) + self.fto[1], true_fun, false_fun)

                self._grating_type_assigned = gear

            # if (self.ucell.shape[1] == 1) and (self.pol in (0, 1)) and (self.phi % (2 * np.pi) == 0):
            #     self._grating_type_assigned = 0  # 1D TE and TM only
            else:
                self._grating_type_assigned = 1  # else

        elif isinstance(self.ucell, list):  # Vector
            self.modeling_type_assigned = 1
            self.grating_type_assigned = 1

    @property
    def grating_type_assigned(self):
        return self._grating_type_assigned

    @grating_type_assigned.setter
    def grating_type_assigned(self, grating_type_assigned):
        self._grating_type_assigned = grating_type_assigned

    def _solve(self, wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all):

        # def false_fun(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all):
        #     de_ri, de_ti, layer_info_list, T1 = self.solve_1d(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)
        #     return de_ri, de_ti, layer_info_list, T1
        #
        # def true_fun(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all):
        #     de_ri, de_ti, layer_info_list, T1 = self.solve_2d(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)
        #     return de_ri, de_ti, layer_info_list, T1
        #
        # de_ri, de_ti, layer_info_list, T1 = jax.lax.cond(self._grating_type_assigned, true_fun, false_fun, wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)

        # if self._grating_type_assigned == 0:
        #     de_ri, de_ti, layer_info_list, T1 = self.solve_1d(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)
        # else:
        #     de_ri, de_ti, layer_info_list, T1 = self.solve_2d(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)

        # In JAXMeent, 1D TE TM are turned off for jit compilation.
        de_ri, de_ti, layer_info_list, T1 = self.solve_2d(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)

        return de_ri, de_ti, layer_info_list, T1

    @jax_device_set
    def solve(self, wavelength, e_conv_all, o_e_conv_all):
        de_ri, de_ti, layer_info_list, T1, kx_vector = jax.jit(self._solve)(wavelength, e_conv_all, o_e_conv_all)

        self.layer_info_list = layer_info_list
        self.T1 = T1

        return de_ri, de_ti

    def _conv_solve(self, **kwargs):
        self._assign_modeling_type()

        if self._modeling_type_assigned == 0:  # Raster

            if self.fourier_type == 0:
                epx_conv_all, epy_conv_all, epz_conv_i_all = to_conv_mat_raster_discrete(
                    self.ucell, self.fto[0], self.fto[1], type_complex=self.type_complex,
                    enhanced_dfs=self.enhanced_dfs)

            elif self.fourier_type == 1:
                epx_conv_all, epy_conv_all, epz_conv_i_all = to_conv_mat_raster_continuous(
                    self.ucell, self.fto[0], self.fto[1], type_complex=self.type_complex)
            else:
                raise ValueError("Check 'modeling_type' and 'fourier_type' in 'conv_solve'.")

        elif self._modeling_type_assigned == 1:  # Vector
            ucell_vector = self.modeling_vector_instruction(self.ucell)
            epx_conv_all, epy_conv_all, epz_conv_i_all = to_conv_mat_vector(
                ucell_vector, self.fto[0], self.fto[1], type_complex=self.type_complex)

        else:
            raise ValueError("Check 'modeling_type' and 'fourier_type' in 'conv_solve'.")

        de_ri, de_ti, layer_info_list, T1 = self._solve(self.wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)

        self.layer_info_list = layer_info_list
        self.T1 = T1

        return de_ri, de_ti, layer_info_list, T1

    @jax.jit
    def _conv_solve_jit(self):
        return self._conv_solve()

    @jax_device_set
    def conv_solve(self, **kwargs):
        [setattr(self, k, v) for k, v in kwargs.items()]  # needed for optimization
        if self.fourier_type == 1:
            # print('CFT (fourier_type=1) is not supported for jit-compilation. Using non-jit-compiled method.')
            de_ri, de_ti, layer_info_list, T1 = self._conv_solve()

        else:
            de_ri, de_ti, layer_info_list, T1 = self._conv_solve()
            # de_ri, de_ti, layer_info_list, T1 = self._conv_solve_jit()

        return de_ri, de_ti

    @jax_device_set
    def calculate_field(self, res_x=20, res_y=20, res_z=20):

        kx, ky = self.get_kx_ky_vector(wavelength=self.wavelength)

        if self._grating_type_assigned == 0:
            res_y = 1
            field_cell = field_dist_1d(self.wavelength, kx, self.T1, self.layer_info_list, self.period, self.pol,
                                       res_x=res_x, res_y=res_y, res_z=res_z, type_complex=self.type_complex)
        else:
            field_cell = field_dist_2d(self.wavelength, kx, ky, self.T1, self.layer_info_list, self.period,
                                       res_x=res_x, res_y=res_y, res_z=res_z, type_complex=self.type_complex)

        return field_cell

    def field_plot(self, field_cell):
        field_plot(field_cell, self.pol)

    @partial(jax.jit, static_argnums=(1, 2, 3))
    @jax_device_set
    def conv_solve_field(self, res_x=20, res_y=20, res_z=20, **kwargs):
        [setattr(self, k, v) for k, v in kwargs.items()]  # needed for optimization

        if self.fourier_type == 1:
            print('CFT (fourier_type=1) is not supported with JAX jit-compilation. Use conv_solve_field_no_jit.')
            return None, None, None

        de_ri, de_ti, _, _ = self._conv_solve()
        field_cell = self.calculate_field(res_x, res_y, res_z)
        return de_ri, de_ti, field_cell

    @jax_device_set
    def conv_solve_field_no_jit(self, res_x=20, res_y=20, res_z=20):
        de_ri, de_ti, _, _ = self._conv_solve()
        field_cell = self.calculate_field(res_x, res_y, res_z)
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
