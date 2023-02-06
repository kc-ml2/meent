import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ._base import _BaseRCWA
from .convolution_matrix import to_conv_mat, put_permittivity_in_ucell, read_material_table, \
    to_conv_mat_piecewise_constant
from .field_distribution import field_dist_1d, field_dist_1d_conical, field_dist_2d, field_plot


class RCWAJax(_BaseRCWA):
    def __init__(self, n_I=1., n_II=1., theta=0, phi=0, psi=0,
                 period=(100,),
                 wavelength=900, ucell=None,
                 thickness=None, perturbation=1E-10,
                 mode=1, grating_type=0,
                 pol=0, fourier_order=40,
                 ucell_materials=None,
                 algo='TMM',
                 device='cpu', type_complex=jnp.complex128):

        super().__init__(grating_type, n_I, n_II, theta, phi, psi, pol, fourier_order, period, wavelength,
                         ucell, ucell_materials,
                         thickness, algo, perturbation, device, type_complex)

        self.device = device
        self.mode = mode
        self.type_complex = type_complex

        self.mat_table = read_material_table(type_complex=self.type_complex)
        self.layer_info_list = []

    def _tree_flatten(self):
        children = (self.n_I, self.n_II, self.theta, self.phi, self.psi,
                    self.period, self.wavelength, self.ucell, self.thickness, self.perturbation)
        aux_data = {
            'mode': self.mode,
            'grating_type': self.grating_type,
            'pol': self.pol,
            'fourier_order': self.fourier_order,
            'ucell_materials': self.ucell_materials,
            'algo': self.algo,
            'device': self.device,
            'type_complex': self.type_complex,

        }

        return children, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    @jax.jit
    def solve(self, wavelength, e_conv_all, o_e_conv_all):

        self.get_kx_vector()
        # self.kx_vector = self.get_kx_vector()

        if self.grating_type == 0:
            de_ri, de_ti, layer_info_list, T1 = self.solve_1d(wavelength, e_conv_all, o_e_conv_all)
        elif self.grating_type == 1:
            de_ri, de_ti, layer_info_list, T1 = self.solve_1d_conical(wavelength, e_conv_all, o_e_conv_all)
        elif self.grating_type == 2:
            de_ri, de_ti, layer_info_list, T1 = self.solve_2d(wavelength, e_conv_all, o_e_conv_all)
        else:
            raise ValueError

        self.layer_info_list = layer_info_list
        self.T1 = T1

        return de_ri.real, de_ti.real

    @jax.jit
    def aaa(self, ucell):
        E_conv_all = to_conv_mat(ucell, self.fourier_order, type_complex=self.type_complex)
        o_E_conv_all = to_conv_mat(1 / ucell, self.fourier_order, type_complex=self.type_complex)

        de_ri, de_ti = self.solve(self.wavelength, E_conv_all, o_E_conv_all)
        return de_ri, de_ti

    def run_ucell(self):

        ucell = put_permittivity_in_ucell(self.ucell, self.ucell_materials, self.mat_table, self.wavelength,
                                          type_complex=self.type_complex)

        E_conv_all = to_conv_mat_piecewise_constant(ucell, self.fourier_order, type_complex=self.type_complex)
        o_E_conv_all = to_conv_mat_piecewise_constant(1 / ucell, self.fourier_order, type_complex=self.type_complex)

        de_ri, de_ti = self.solve(self.wavelength, E_conv_all, o_E_conv_all)

        # de_ri, de_ti = self.aaa(ucell)

        return de_ri, de_ti

    def run_ucell_vmap(self):

        ucell = put_permittivity_in_ucell(self.ucell, self.ucell_materials, self.mat_table, self.wavelength,
                                          type_complex=self.type_complex)
        E_conv_all = to_conv_mat(ucell, self.fourier_order, type_complex=self.type_complex)
        o_E_conv_all = to_conv_mat(1 / ucell, self.fourier_order, type_complex=self.type_complex)

        self.wavelength = 901
        ucell1 = put_permittivity_in_ucell(self.ucell, self.ucell_materials, self.mat_table, self.wavelength,
                                          type_complex=self.type_complex)
        E_conv_all1 = to_conv_mat(ucell, self.fourier_order, type_complex=self.type_complex)
        o_E_conv_all1 = to_conv_mat(1 / ucell, self.fourier_order, type_complex=self.type_complex)

        self.wavelength = 902
        ucell2 = put_permittivity_in_ucell(self.ucell, self.ucell_materials, self.mat_table, self.wavelength,
                                          type_complex=self.type_complex)
        E_conv_all2 = to_conv_mat(ucell, self.fourier_order, type_complex=self.type_complex)
        o_E_conv_all2 = to_conv_mat(1 / ucell, self.fourier_order, type_complex=self.type_complex)

        self.wavelength = 903
        ucell3 = put_permittivity_in_ucell(self.ucell, self.ucell_materials, self.mat_table, self.wavelength,
                                          type_complex=self.type_complex)
        E_conv_all3 = to_conv_mat(ucell, self.fourier_order, type_complex=self.type_complex)
        o_E_conv_all3 = to_conv_mat(1 / ucell, self.fourier_order, type_complex=self.type_complex)

        a = jnp.array([900, 901, 902, 903])
        b = jnp.array([E_conv_all, E_conv_all1, E_conv_all2, E_conv_all3])
        c = jnp.array([o_E_conv_all, o_E_conv_all1, o_E_conv_all2, o_E_conv_all3])

        de_ri, de_ti = jax.vmap(self.solve)(a, b, c)

        return de_ri, de_ti

    def run_ucell_pmap(self):

        ucell = put_permittivity_in_ucell(self.ucell, self.ucell_materials, self.mat_table, self.wavelength,
                                          type_complex=self.type_complex)
        E_conv_all = to_conv_mat(ucell, self.fourier_order, type_complex=self.type_complex)
        o_E_conv_all = to_conv_mat(1 / ucell, self.fourier_order, type_complex=self.type_complex)

        self.wavelength = 901
        ucell1 = put_permittivity_in_ucell(self.ucell, self.ucell_materials, self.mat_table, self.wavelength,
                                          type_complex=self.type_complex)
        E_conv_all1 = to_conv_mat(ucell, self.fourier_order, type_complex=self.type_complex)
        o_E_conv_all1 = to_conv_mat(1 / ucell, self.fourier_order, type_complex=self.type_complex)

        self.wavelength = 902
        ucell2 = put_permittivity_in_ucell(self.ucell, self.ucell_materials, self.mat_table, self.wavelength,
                                          type_complex=self.type_complex)
        E_conv_all2 = to_conv_mat(ucell, self.fourier_order, type_complex=self.type_complex)
        o_E_conv_all2 = to_conv_mat(1 / ucell, self.fourier_order, type_complex=self.type_complex)

        self.wavelength = 903
        ucell3 = put_permittivity_in_ucell(self.ucell, self.ucell_materials, self.mat_table, self.wavelength,
                                          type_complex=self.type_complex)
        E_conv_all3 = to_conv_mat(ucell, self.fourier_order, type_complex=self.type_complex)
        o_E_conv_all3 = to_conv_mat(1 / ucell, self.fourier_order, type_complex=self.type_complex)

        a = jnp.array([900, 901, 902, 903])
        b = jnp.array([E_conv_all, E_conv_all1, E_conv_all2, E_conv_all3])
        c = jnp.array([o_E_conv_all, o_E_conv_all1, o_E_conv_all2, o_E_conv_all3])

        de_ri, de_ti = jax.pmap(self.solve)(a, b, c)

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


if __name__ == '__main__':
    pass
