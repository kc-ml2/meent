import time
import numpy as np

from ._base import _BaseRCWA
from .convolution_matrix import to_conv_mat_piecewise_constant, put_permittivity_in_ucell, read_material_table
from .field_distribution import field_dist_1d, field_dist_1d_conical, field_dist_2d, field_plot_zx


class RCWANumpy(_BaseRCWA):
    def __init__(self, mode=0, grating_type=0, n_I=1., n_II=1., theta=0, phi=0, psi=0, fourier_order=40, period=(100,),
                 wavelength=900, pol=0, patterns=None, ucell=None, ucell_materials=None,
                 thickness=None, algo='TMM', perturbation=1E-10,
                 device='cpu', type_complex=np.complex128):

        super().__init__(grating_type, n_I, n_II, theta, phi, psi, fourier_order, period, wavelength, pol, patterns,
                         ucell, ucell_materials,
                         thickness, algo, perturbation, device, type_complex)

        self.device = 'cpu'
        self.mode = mode
        self.type_complex = type_complex

        self.mat_table = read_material_table()
        self.layer_info_list = []

    def put_permittivity_in_ucell(self):
        ucell = put_permittivity_in_ucell(self.ucell, self.ucell_materials, self.mat_table, self.wavelength,
                                          type_complex=self.type_complex)
        return ucell

    def to_conv_mat(self, ucell):
        E_conv_all = to_conv_mat_piecewise_constant(ucell, self.fourier_order, type_complex=self.type_complex)
        return E_conv_all

    def solve(self, wavelength, e_conv_all, o_e_conv_all):

        self.get_kx_vector()

        if self.grating_type == 0:
            de_ri, de_ti = self.solve_1d(wavelength, e_conv_all, o_e_conv_all)
        elif self.grating_type == 1:
            de_ri, de_ti = self.solve_1d_conical(wavelength, e_conv_all, o_e_conv_all)
        elif self.grating_type == 2:
            de_ri, de_ti = self.solve_2d(wavelength, e_conv_all, o_e_conv_all)
        else:
            raise ValueError

        return de_ri.real, de_ti.real

    def run_ucell(self):

        ucell = put_permittivity_in_ucell(self.ucell, self.ucell_materials, self.mat_table, self.wavelength,
                                          type_complex=self.type_complex)

        E_conv_all = to_conv_mat_piecewise_constant(ucell, self.fourier_order, type_complex=self.type_complex)
        o_E_conv_all = to_conv_mat_piecewise_constant(1 / ucell, self.fourier_order, type_complex=self.type_complex)

        # apply to other backends (removing wavelength arg)
        de_ri, de_ti = self.solve(E_conv_all, o_E_conv_all)

        return de_ri, de_ti

    def calculate_field(self, resolution=None, plot=True):

        if self.grating_type == 0:
            resolution = [100, 1, 100] if not resolution else resolution
            field_cell = field_dist_1d(self.wavelength, self.n_I, self.theta, self.fourier_order, self.T1,
                                       self.layer_info_list, self.period, self.pol, resolution=resolution,
                                       type_complex=self.type_complex)
        elif self.grating_type == 1:
            resolution = [100, 1, 100] if not resolution else resolution
            field_cell = field_dist_1d_conical(self.wavelength, self.n_I, self.theta, self.phi, self.fourier_order,
                                               self.T1,
                                               self.layer_info_list, self.period, resolution=resolution,
                                               type_complex=self.type_complex)

        else:
            resolution = [100, 100, 100] if not resolution else resolution
            t0 = time.time()
            field_cell = field_dist_2d(self.wavelength, self.n_I, self.theta, self.phi, self.fourier_order, self.T1,
                                       self.layer_info_list, self.period, resolution=resolution,
                                       type_complex=self.type_complex)
            print(time.time() - t0)
        if plot:
            field_plot_zx(field_cell, self.pol)

        return field_cell
