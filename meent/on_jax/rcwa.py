import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ._base import _BaseRCWA
from .convolution_matrix import to_conv_mat_discrete, put_permittivity_in_ucell, read_material_table, \
    to_conv_mat_continuous
from .field_distribution import field_dist_1d, field_dist_1d_conical, field_dist_2d, field_plot


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
                 thickness=None,
                 mode=1,
                 grating_type=0,
                 pol=0,
                 fourier_order=40,
                 ucell_materials=None,
                 algo='TMM',
                 perturbation=1E-10,
                 device='cpu',
                 type_complex=jnp.complex128,
                 fft_type=0,
                 improve_dft=True,
                 ):

        super().__init__(grating_type=grating_type, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi, pol=pol,
                         fourier_order=fourier_order, period=period, wavelength=wavelength,
                         ucell=ucell, ucell_materials=ucell_materials,
                         thickness=thickness, algo=algo, perturbation=perturbation,
                         device=device, type_complex=type_complex,)

        self.mode = mode
        self.device = device
        self.type_complex = type_complex
        self.fft_type = fft_type
        self.improve_dft = improve_dft

        self.mat_table = read_material_table(type_complex=self.type_complex)
        self.layer_info_list = []

    def _tree_flatten(self):
        children = (self.n_I, self.n_II, self.theta, self.phi, self.psi,
                    self.period, self.wavelength, self.ucell, self.thickness)
        aux_data = {
            'mode': self.mode,
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

    @jax.jit
    def solve(self, wavelength, e_conv_all, o_e_conv_all):
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

    @jax.jit
    def conv_solve(self, ucell):
        E_conv_all = to_conv_mat_discrete(ucell, self.fourier_order, type_complex=self.type_complex, improve_dft=self.improve_dft)
        o_E_conv_all = to_conv_mat_discrete(1 / ucell, self.fourier_order, type_complex=self.type_complex, improve_dft=self.improve_dft)
        de_ri, de_ti, layer_info_list, T1, kx_vector = self.solve(self.wavelength, E_conv_all, o_E_conv_all)
        return de_ri, de_ti

    @jax.jit
    def conv_solve_spectrum(self, ucell):
        E_conv_all = to_conv_mat_discrete(ucell, self.fourier_order, type_complex=self.type_complex, improve_dft=self.improve_dft)
        o_E_conv_all = to_conv_mat_discrete(1 / ucell, self.fourier_order, type_complex=self.type_complex, improve_dft=self.improve_dft)
        de_ri, de_ti, layer_info_list, T1, kx_vector = self.solve(self.wavelength, E_conv_all, o_E_conv_all)
        return de_ri, de_ti

    def run_ucell(self):
        ucell = put_permittivity_in_ucell(self.ucell, self.ucell_materials, self.mat_table, self.wavelength,
                                          type_complex=self.type_complex)
        if self.fft_type == 0:
            E_conv_all = to_conv_mat_discrete(ucell, self.fourier_order, type_complex=self.type_complex, improve_dft=self.improve_dft)
            o_E_conv_all = to_conv_mat_discrete(1 / ucell, self.fourier_order, type_complex=self.type_complex, improve_dft=self.improve_dft)
        elif self.fft_type == 1:
            E_conv_all = to_conv_mat_continuous(ucell, self.fourier_order, type_complex=self.type_complex)
            o_E_conv_all = to_conv_mat_continuous(1 / ucell, self.fourier_order, type_complex=self.type_complex)
        else:
            raise ValueError

        de_ri, de_ti, layer_info_list, T1, kx_vector = self.solve(self.wavelength, E_conv_all, o_E_conv_all)

        self.layer_info_list = layer_info_list
        self.T1 = T1
        self.kx_vector = kx_vector

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

        de_ri = np.array(de_ri)
        de_ti = np.array(de_ti)
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

    # def generate_spectrum(self, wl_list):
    #     ucell = deepcopy(self.ucell)
    #     spectrum_ri_pmap = np.zeros(wl_list.shape)
    #     spectrum_ti_pmap = np.zeros(wl_list.shape)
    #     for i, wavelength in enumerate(range(counter)):
    #         b = i * num_device
    #         de_ri_pmap, de_ti_pmap = jax.pmap(loop_wavelength)(wavelength_list[b:b + num_device],
    #                                                            mat_pmtvy_interp[b:b + num_device])
    #
    #         spectrum_ri_pmap[b:b + num_device] = de_ri_pmap.sum(axis=(1, 2))
    #         spectrum_ti_pmap[b:b + num_device] = de_ti_pmap.sum(axis=(1, 2))
    #
    #     return spectrum_ri_pmap, spectrum_ti_pmap


if __name__ == '__main__':
    pass
