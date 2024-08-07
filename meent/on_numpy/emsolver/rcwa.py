import time
import numpy as np

from ._base import _BaseRCWA
from .convolution_matrix import to_conv_mat_raster_continuous, to_conv_mat_raster_discrete, to_conv_mat_vector
from .field_distribution import field_plot, field_dist_1d, field_dist_2d


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
                 grating_type=None,  # TODO: remove
                 modeling_type=None,
                 pol=0.,
                 fto=(0, 0),
                 ucell_materials=None,
                 connecting_algo='TMM',
                 perturbation=1E-20,
                 device='cpu',
                 type_complex=np.complex128,
                 fourier_type=None,  # 0 DFS, 1 EFS, 2 CFS
                 enhanced_dfs=True,
                 **kwargs,
                 ):

        super().__init__(n_top=n_top, n_bot=n_bot, theta=theta, phi=phi, psi=psi, pol=pol,
                         fto=fto, period=period, wavelength=wavelength,
                         thickness=thickness, connecting_algo=connecting_algo, perturbation=perturbation,
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

        # grating type setting
        if self.grating_type is None:
            if (type(self.ucell) is np.ndarray and self.ucell.shape[1] == 1) and (self.pol in (0, 1)) and (self.phi % (2*np.pi) == 0):
                self._grating_type_assigned = 0
            else:
                self._grating_type_assigned = 1  # TODO
        else:
            self._grating_type_assigned = self.grating_type

        # modeling type setting
        if self.modeling_type is None:
            if self.ucell_info_list is None:
                self._modeling_type_assigned = 0
            elif self.ucell is None:
                self._modeling_type_assigned = 1
                self._grating_type_assigned = 1
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

        if self._ucell is not None:
            self._modeling_type_assigned = 0  # Raster type

            if self._ucell.shape[1] == 1:  # TODO
                self._grating_type_assigned = 0
            else:
                self._grating_type_assigned = 1

    @property
    def ucell_info_list(self):
        return self._ucell_info_list

    @ucell_info_list.setter
    def ucell_info_list(self, ucell_info_list):
        self._ucell_info_list = ucell_info_list
        if ucell_info_list is not None:
            self._modeling_type_assigned = 1  # Vector type
            self._grating_type_assigned = 1

    def _solve(self, wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all):

        if self._grating_type_assigned == 0:
            de_ri, de_ti, layer_info_list, T1 = self.solve_1d(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)
        else:
            de_ri, de_ti, layer_info_list, T1 = self.solve_2d(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)

        return de_ri, de_ti, layer_info_list, T1

    def solve(self, wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all):

        de_ri, de_ti, layer_info_list, T1 = self._solve(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)

        self.layer_info_list = layer_info_list
        self.T1 = T1

        return de_ri, de_ti

    def conv_solve(self, **kwargs):
        # [setattr(self, k, v) for k, v in kwargs.items()]  # no need in npmeent

        if self._modeling_type_assigned == 0:  # Raster

            if self.fourier_type == 0:
                epx_conv_all, epy_conv_all, epz_conv_i_all = to_conv_mat_raster_discrete(
                    self.ucell, self.fto[0], self.fto[1], type_complex=self.type_complex,
                    enhanced_dfs=self.enhanced_dfs)

            # if self.fourier_type == 0:
            #     enhance = False
            #     epx_conv_all, epy_conv_all, epz_conv_i_all = to_conv_mat_raster_discrete(
            #         self.ucell, self.fto[0], self.fto[1], type_complex=self.type_complex,
            #         enhanced_dfs=enhance)
            #
            # elif (self.fourier_type == 1) or (self.fourier_type is None):
            #     enhance = True
            #     epx_conv_all, epy_conv_all, epz_conv_i_all = to_conv_mat_raster_discrete(
            #         self.ucell, self.fto[0], self.fto[1], type_complex=self.type_complex,
            #         enhanced_dfs=enhance)

            elif self.fourier_type == 1:
                epx_conv_all, epy_conv_all, epz_conv_i_all = to_conv_mat_raster_continuous(
                    self.ucell, self.fto[0], self.fto[1], type_complex=self.type_complex)
            else:
                raise ValueError("Check 'modeling_type' and 'fourier_type' in 'conv_solve'.")

        elif self._modeling_type_assigned == 1:  # Vector
            epx_conv_all, epy_conv_all, epz_conv_i_all = to_conv_mat_vector(
                self.ucell_info_list, self.fto[0], self.fto[1], type_complex=self.type_complex)

        else:
            raise ValueError("Check 'modeling_type' and 'fourier_type' in 'conv_solve'.")

        de_ri, de_ti, layer_info_list, T1 = self._solve(self.wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)

        self.layer_info_list = layer_info_list
        self.T1 = T1

        return de_ri, de_ti

    def calculate_field(self, res_x=20, res_y=20, res_z=20):
        # TODO: change res_ to accept array of points.
        kx, ky = self.get_kx_ky_vector(wavelength=self.wavelength)

        if self._grating_type_assigned == 0:
            res_y = 1
            field_cell = field_dist_1d(self.wavelength, kx, self.T1, self.layer_info_list, self.period, self.pol,
                                       res_x=res_x, res_y=res_y, res_z=res_z, type_complex=self.type_complex)
        else:
            field_cell = field_dist_2d(self.wavelength, kx, ky, self.T1, self.layer_info_list, self.period,
                                       res_x=res_x, res_y=res_y, res_z=res_z, type_complex=self.type_complex)

        return field_cell

    def conv_solve_field(self, res_x=20, res_y=20, res_z=20):
        de_ri, de_ti = self.conv_solve()
        field_cell = self.calculate_field(res_x, res_y, res_z)
        return de_ri, de_ti, field_cell

    def field_plot(self, field_cell):
        field_plot(field_cell, self.pol)
