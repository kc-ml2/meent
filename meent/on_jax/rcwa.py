import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ._base import _BaseRCWA
from .convolution_matrix import to_conv_mat, put_permittivity_in_ucell, read_material_table
from .field_distribution import field_dist_1d, field_dist_1d_conical, field_dist_2d, field_plot


class RCWAJax(_BaseRCWA):
    def __init__(self, mode=0, grating_type=0, n_I=1., n_II=1., theta=0, phi=0, psi=0, fourier_order=40, period=(100,),
                 wavelength=jnp.linspace(900, 900, 1), pol=0, patterns=None, ucell=None, ucell_materials=None,
                 thickness=None, algo='TMM',
                 device='cpu', type_complex=np.complex128):

        super().__init__(grating_type, n_I, n_II, theta, phi, psi, fourier_order, period, wavelength, pol, patterns, ucell, ucell_materials,
                         thickness, algo, device, type_complex)
        self.mode = mode
        self.spectrum_r, self.spectrum_t = None, None
        # self.init_spectrum_array()
        self.mat_table = read_material_table()
        self.layer_info_list = []

    def solve(self, wavelength, e_conv_all, o_e_conv_all):
        # TODO: perturbation
        self.perturbation = 1E-20

        # TODO: move to _base?
        def get_kx_vector(perturbation=self.perturbation):

            k0 = 2 * jnp.pi / self.wavelength
            fourier_indices = jnp.arange(-self.fourier_order, self.fourier_order + 1)
            if self.grating_type == 0:
                kx_vector = k0 * (self.n_I * jnp.sin(self.theta) - fourier_indices * (self.wavelength / self.period[0])
                                  ).astype(self.type_complex)
            else:
                kx_vector = k0 * (self.n_I * jnp.sin(self.theta) * jnp.cos(self.phi) - fourier_indices * (self.wavelength / self.period[0])
                                  ).astype(self.type_complex)

            idx = jnp.nonzero(kx_vector == 0)[0]
            if len(idx):
                # TODO: need imaginary part?
                # TODO: make imaginary part sign consistent
                kx_vector = kx_vector.at[idx].set(perturbation)
                print('varphi divide by 0: adding perturbation')

            self.kx_vector = kx_vector
            return kx_vector

        # TODO: handle uniform layer

        t0=time.time()
        get_kx_vector()

        if self.grating_type == 0:
            solve_1d = jax.jit(self.solve_1d)
            de_ri, de_ti, layer_info_list, T1 = solve_1d(wavelength, e_conv_all, o_e_conv_all)
        elif self.grating_type == 1:
            solve_1d_conical = jax.jit(self.solve_1d_conical)
            de_ri, de_ti, layer_info_list, T1 = solve_1d_conical(wavelength, e_conv_all, o_e_conv_all)

        elif self.grating_type == 2:
            solve_2d = jax.jit(self.solve_2d)
            de_ri, de_ti, layer_info_list, T1 = solve_2d(wavelength, e_conv_all, o_e_conv_all)
        else:
            raise ValueError

        self.layer_info_list = layer_info_list
        self.T1 = T1

        print('solve time', time.time() - t0)

        return de_ri.real, de_ti.real

    def run_ucell(self):

        ucell = put_permittivity_in_ucell(self.ucell, self.ucell_materials, self.mat_table, self.wavelength,
                                          type_complex=self.type_complex)

        E_conv_all = to_conv_mat(ucell, self.fourier_order, type_complex=self.type_complex)
        o_E_conv_all = to_conv_mat(1 / ucell, self.fourier_order, type_complex=self.type_complex)

        de_ri, de_ti = self.solve(self.wavelength, E_conv_all, o_E_conv_all)

        return de_ri, de_ti

    def calculate_field(self, resolution=None, plot=True):

        if self.grating_type == 0:
            resolution = [100, 1, 100] if not resolution else resolution
            field_cell = field_dist_1d(self.wavelength, self.kx_vector, self.n_I, self.theta, self.fourier_order, self.T1,
                                       self.layer_info_list, self.period, self.pol, resolution=resolution,
                                       type_complex=self.type_complex)
        elif self.grating_type == 1:
            resolution = [100, 1, 100] if not resolution else resolution
            field_cell = field_dist_1d_conical(self.wavelength, self.kx_vector, self.n_I, self.theta, self.phi, self.fourier_order, self.T1,
                                               self.layer_info_list, self.period, resolution=resolution,
                                               type_complex=self.type_complex)

        else:
            resolution = [10, 10, 10] if not resolution else resolution
            field_cell = field_dist_2d(self.wavelength, self.kx_vector, self.n_I, self.theta, self.phi, self.fourier_order, self.T1,
                                       self.layer_info_list, self.period, resolution=resolution,
                                       type_complex=self.type_complex)
        if plot:
            field_plot(field_cell, self.pol)
        return field_cell

    # def calculate_field_jax(self, resolution=None, plot=True):
    #
    #     ucell = put_permittivity_in_ucell(self.ucell, self.ucell_materials, self.mat_table, self.wavelength)
    #     e_conv_all = to_conv_mat(ucell, self.fourier_order)
    #
    #     o_e_conv_all = to_conv_mat(1 / ucell, self.fourier_order)
    #
    #     field_cell = self.bb(e_conv_all, o_e_conv_all, resolution=resolution, plot=plot)
    #     if plot:
    #         field_plot(field_cell, self.pol, zx=True, yx=False)
    #
    #     return e_conv_all, o_e_conv_all
    #
    # @partial(jax.jit, static_argnums=(0, 3, 4))
    # def bb(self, e_conv_all, o_e_conv_all, resolution=None, plot=True):
    #     print('bb compile')
    #     t0= time.time()
    #     # self.solve(self.wavelength, e_conv_all, o_e_conv_all, jit=True)
    #     # self.solve_2d(self.wavelength, e_conv_all, o_e_conv_all)
    #     if self.grating_type == 0:
    #         resolution = (100, 1, 100) if not resolution else resolution
    #         field_cell = field_dist_1d(self.wavelength, self.n_I, self.theta, self.fourier_order, self.T1,
    #                             self.layer_info_list, self.period, self.pol, resolution=resolution)
    #     elif self.grating_type == 1:
    #         resolution = (100, 1, 100) if not resolution else resolution
    #         field_cell = field_dist_1d_conical(self.wavelength, self.n_I, self.theta, self.phi, self.fourier_order, self.T1,
    #                             self.layer_info_list, self.period, resolution=resolution)
    #     else:
    #         resolution = (20, 20, 20) if not resolution else resolution
    #         field_cell = field_dist_2d(self.wavelength, self.n_I, self.theta, self.phi, self.fourier_order, self.T1,
    #                                    self.layer_info_list, self.period, resolution=resolution)
    #     print('bb', time.time() -t0)
    #     # if plot:
    #     #     field_plot(field_cell, self.pol, zx=True, yx=False)
    #     return field_cell


if __name__ == '__main__':
    pass