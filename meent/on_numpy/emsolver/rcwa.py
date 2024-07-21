import time
import numpy as np

from ._base import _BaseRCWA
from .convolution_matrix import to_conv_mat_raster_continuous, to_conv_mat_raster_discrete, to_conv_mat_vector
from .field_distribution import field_dist_1d_vectorized_ji, field_dist_1d_conical_vectorized_ji, field_dist_2d_vectorized_ji, field_plot, field_dist_1d_vanilla, \
    field_dist_1d_vectorized_kji, field_dist_1d_conical_vanilla, field_dist_1d_conical_vectorized_kji, \
    field_dist_2d_vectorized_kji, field_dist_2d_vanilla


class RCWANumpy(_BaseRCWA):
    def __init__(self,
                 n_top=1.,
                 n_bot=1.,
                 theta=0.,
                 phi=0.,
                 psi=None,
                 period=(100., 100.),
                 wavelength=900.,
                 ucell=None,
                 ucell_info_list=None,
                 thickness=(0., ),
                 backend=0,
                 grating_type=None,
                 modeling_type=None,
                 pol=0.,
                 fto=(0, 0),
                 ucell_materials=None,
                 stitching_algo='TMM',
                 perturbation=1E-20,
                 device='cpu',
                 type_complex=np.complex128,
                 fourier_type=None,  # 0 DFS, 1 EFS, 2 CFS
                 enhanced_dfs=True,
                 **kwargs,
                 ):

        super().__init__(n_top=n_top, n_bot=n_bot, theta=theta, phi=phi, psi=psi, pol=pol,
                         fto=fto, period=period, wavelength=wavelength,
                         thickness=thickness, stitching_algo=stitching_algo, perturbation=perturbation,
                         device=device, type_complex=type_complex, )

        self.ucell = ucell
        self.ucell_materials = ucell_materials
        self.ucell_info_list = ucell_info_list

        self.backend = backend
        self.modeling_type = modeling_type
        self._modeling_type_assigned = None
        self.grating_type = grating_type
        self._grating_type_assigned = None
        self.fourier_type = fourier_type
        self.enhanced_dfs = enhanced_dfs

        self.layer_info_list = []

        # grating type setting
        if self.grating_type is None:
            if self.ucell.shape[1] == 1:
                if (self.pol in (0, 1)) and (self.phi % (2*np.pi) == 0):
                    self._grating_type_assigned = 0
                else:
                    self._grating_type_assigned = 1
            else:
                self._grating_type_assigned = 2
        else:
            self._grating_type_assigned = self.grating_type

        # modeling type setting
        if self.modeling_type is None:
            if self.ucell_info_list is None:
                self._modeling_type_assigned = 0
            elif self.ucell is None:
                self._modeling_type_assigned = 1
            else:
                raise ValueError('Define "modeling_type" in "call_mee" function.')
        else:
            self._modeling_type_assigned = self.modeling_type

    @property
    def ucell(self):
        return self._ucell

    @ucell.setter
    def ucell(self, ucell):
        if isinstance(ucell, np.ndarray):
            if ucell.dtype in (np.int64, np.float64, np.int32, np.float32):
                dtype = self.type_float
            elif ucell.dtype in (np.complex128, np.complex64):
                dtype = self.type_complex
            else:
                raise ValueError
            self._ucell = np.array(ucell, dtype=dtype)
        elif ucell is None:
            self._ucell = ucell
        else:
            raise ValueError

    def _solve(self, wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all):

        if self._grating_type_assigned == 0:
            de_ri, de_ti, layer_info_list, T1 = self.solve_1d(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)
        elif self._grating_type_assigned == 1:
            de_ri, de_ti, layer_info_list, T1 = self.solve_1d_conical(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)
        elif self._grating_type_assigned == 2:
            de_ri, de_ti, layer_info_list, T1 = self.solve_2d(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)
        else:
            raise ValueError

        return de_ri, de_ti, layer_info_list, T1

    def solve(self, wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all):

        de_ri, de_ti, layer_info_list, T1 = self._solve(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)

        self.layer_info_list = layer_info_list
        self.T1 = T1

        return de_ri, de_ti

    def conv_solve(self, **kwargs):
        # [setattr(self, k, v) for k, v in kwargs.items()]  # no need in npmeent

        if self._modeling_type_assigned == 0 and self.fourier_type in (0, 1):
            epx_conv_all, epy_conv_all, epz_conv_i_all = to_conv_mat_raster_discrete(
                self.ucell, self.fto[0], self.fto[1], type_complex=self.type_complex, enhanced_dfs=self.fourier_type)
        elif self._modeling_type_assigned == 0 and self.fourier_type == 2:
            epx_conv_all, epy_conv_all, epz_conv_i_all = to_conv_mat_raster_continuous(
                self.ucell, self.fto[0], self.fto[1], type_complex=self.type_complex)
        elif self._modeling_type_assigned == 1:
            epx_conv_all, epy_conv_all, epz_conv_i_all = to_conv_mat_vector(
                self.ucell_info_list, self.fto[0], self.fto[1], type_complex=self.type_complex)
        else:
            raise ValueError("Check 'modeling_type' and 'fourier_type'.")

        de_ri, de_ti, layer_info_list, T1 = self._solve(self.wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)

        self.layer_info_list = layer_info_list
        self.T1 = T1

        return de_ri, de_ti

    def calculate_field(self, res_x=20, res_y=20, res_z=20, field_algo=2):
        kx, ky = self.get_kx_ky_vector(wavelength=self.wavelength)
        if self._grating_type_assigned == 0:
            res_y = 1
            if field_algo == 0:
                field_cell = field_dist_1d_vanilla(self.wavelength, kx,
                                                   self.T1, self.layer_info_list, self.period, self.pol,
                                                   res_x=res_x, res_y=res_y, res_z=res_z, type_complex=self.type_complex)
            elif field_algo == 1:
                field_cell = field_dist_1d_vectorized_ji(self.wavelength, kx, self.T1, self.layer_info_list,
                                                         self.period, self.pol, res_x=res_x, res_y=res_y, res_z=res_z,
                                                         type_complex=self.type_complex)
            elif field_algo == 2:
                field_cell = field_dist_1d_vectorized_kji(self.wavelength, kx, self.T1,
                                                          self.layer_info_list, self.period, self.pol,
                                                          res_x=res_x, res_y=res_y, res_z=res_z, type_complex=self.type_complex)
            else:
                raise ValueError
        elif self._grating_type_assigned == 1:
            res_y = 1
            if field_algo == 0:
                field_cell = field_dist_1d_conical_vanilla(self.wavelength, kx, ky, self.T1, self.layer_info_list, self.period,
                                                           res_x=res_x, res_y=res_y, res_z=res_z, type_complex=self.type_complex)
            elif field_algo == 1:
                field_cell = field_dist_1d_conical_vectorized_ji(self.wavelength, kx, ky, self.T1, self.layer_info_list, self.period,
                                                                 res_x=res_x, res_y=res_y, res_z=res_z, type_complex=self.type_complex)
            elif field_algo == 2:
                field_cell = field_dist_1d_conical_vectorized_kji(self.wavelength,kx, ky, self.T1, self.layer_info_list, self.period,
                                                                  res_x=res_x, res_y=res_y, res_z=res_z, type_complex=self.type_complex)
            else:
                raise ValueError
        elif self._grating_type_assigned == 2:
            if field_algo == 0:
                field_cell = field_dist_2d_vanilla(self.wavelength, kx, ky, self.T1, self.layer_info_list, self.period,
                                                   res_x=res_x, res_y=res_y, res_z=res_z, type_complex=self.type_complex)
            elif field_algo == 1:
                field_cell = field_dist_2d_vectorized_ji(self.wavelength, kx, ky, self.T1, self.layer_info_list,
                                                         self.period, res_x=res_x, res_y=res_y, res_z=res_z,
                                                         type_complex=self.type_complex)
            elif field_algo == 2:
                field_cell = field_dist_2d_vectorized_kji(self.wavelength, kx, ky, self.T1, self.layer_info_list,
                                                          self.period, res_x=res_x, res_y=res_y, res_z=res_z,
                                                          type_complex=self.type_complex)
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

        print('gap(1-0): ', np.linalg.norm(field_cell1 - field_cell0))
        print('gap(2-1): ', np.linalg.norm(field_cell2 - field_cell1))
        print('gap(0-2): ', np.linalg.norm(field_cell0 - field_cell2))

        return field_cell0, field_cell1, field_cell2
