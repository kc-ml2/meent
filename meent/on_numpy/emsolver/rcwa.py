import numpy as np

from ._base import _BaseRCWA
from .convolution_matrix import to_conv_mat_raster_continuous, to_conv_mat_raster_discrete, to_conv_mat_vector
from .field_distribution import field_plot, field_dist_1d, field_dist_2d


class ResultNumpy:
    def __init__(self, psi, R_s, R_p, de_ri, de_ri_s, de_ri_p, de_ti, de_ti_s, de_ti_p):
        self.psi = psi
        self.R_s = R_s
        self.R_p = R_p
        self.de_ri = de_ri
        self.de_ri_s = de_ri_s
        self.de_ri_p = de_ri_p

        self.de_ti = de_ti
        self.de_ti_s = de_ti_s
        self.de_ti_p = de_ti_p

        # TE incidence only
        # self._R_s_TEinc = None
        # self._de_ri_TEinc = None
        #
        # # TM incidence only
        # self._R_p_TMinc = None
        # self._de_ri_TMinc = None

    @property
    def R_s_normalized(self):
        return self.R_s / np.sin(self.psi)

    @property
    def R_p_normalized(self):
        return self.R_p / np.cos(self.psi) / 1j

    @property
    def de_ri_s_normalized(self):
        if self.psi == 0:
            return np.zeros(self.de_ri_s.shape)
        else:
            return self.de_ri_s / np.sin(self.psi)**2

    @property
    def de_ti_p_normalized(self):
        if self.psi == np.pi/2:
            return np.zeros(self.de_ri_p.shape)
        else:
            return self.de_ti_p / np.cos(self.psi)**2




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
                         device=device, type_complex=type_complex, )

        self._modeling_type_assigned = None
        self.grating_type_assigned = None

        self.ucell = ucell
        self.ucell_materials = ucell_materials

        self.backend = backend
        self.fourier_type = fourier_type
        self.enhanced_dfs = enhanced_dfs

    @property
    def ucell(self):
        return self._ucell

    @ucell.setter
    def ucell(self, ucell):

        if isinstance(ucell, np.ndarray):  # Raster
            self._modeling_type_assigned = 0
            if ucell.dtype in (np.int64, np.float64, np.int32, np.float32):
                dtype = self.type_float
            elif ucell.dtype in (np.complex128, np.complex64):
                dtype = self.type_complex
            else:
                raise ValueError
            # self._ucell = np.array(ucell, dtype=dtype)
            self._ucell = ucell.astype(dtype)

        elif type(ucell) is list:  # Vector
            self._modeling_type_assigned = 1
            self._ucell = ucell
        elif ucell is None:
            self._ucell = ucell
        else:
            raise ValueError

    @property
    def modeling_type_assigned(self):
        return self._modeling_type_assigned

    # @modeling_type_assigned.setter
    # def modeling_type_assigned(self, modeling_type_assigned):
    #     self._modeling_type_assigned = modeling_type_assigned

    def _assign_grating_type(self):

        if self.modeling_type_assigned == 0:
            if self.ucell.shape[1] == 1:
                # TODO: isreal or iscomplexobj
                # if (self.pol in (0,1,)) and (self.phi is not None) and (self.fto[1] == 0):
                if (self.pol in (0,1,)) and (np.isreal(self.phi)) and (self.phi.real % (2 * np.pi) == 0) and (self.fto[1] == 0):
                # if (self.pol in (0, 1)) and (not np.iscomplexobj(self.phi)) and (self.phi.real % (2 * np.pi) == 0) and (self.fto[1] == 0):
                    self._grating_type_assigned = 0  # 1D TE and TM only
                else:
                    self._grating_type_assigned = 1  # 1D conical
            else:
                self._grating_type_assigned = 2  # else

        elif self.modeling_type_assigned == 1:
            self.grating_type_assigned = 2

    @property
    def grating_type_assigned(self):
        return self._grating_type_assigned

    @grating_type_assigned.setter
    def grating_type_assigned(self, grating_type_assigned):
        self._grating_type_assigned = grating_type_assigned

    # def _solve(self, wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all):
    #
    #     self._assign_grating_type()
    #
    #     if self.grating_type_assigned == 0:
    #         de_ri_s, de_ri_p, de_ti_s, de_ti_p, layer_info_list, T1, R_s, R_p, T_s, T_p = self.solve_1d(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)
    #     else:
    #         de_ri_s, de_ri_p, de_ti_s, de_ti_p, layer_info_list, T1, R_s, R_p, T_s, T_p = self.solve_2d(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)
    #
    #     return de_ri_s, de_ri_p, de_ti_s, de_ti_p, layer_info_list, T1, R_s, R_p, T_s, T_p

    def solve(self, wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all):

        # de_ri_s, de_ri_p, de_ti_s, de_ti_p, layer_info_list, T1, R_s, R_p, T_s, T_p = self._solve(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)

        self._assign_grating_type()

        if self.grating_type_assigned == 0:
            # de_ri_s, de_ri_p, de_ti_s, de_ti_p, layer_info_list, T1, R_s, R_p, T_s, T_p = self.solve_1d(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)
            res = self.solve_1d(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)
        else:
            # de_ri_s, de_ri_p, de_ti_s, de_ti_p, layer_info_list, T1, R_s, R_p, T_s, T_p = self.solve_2d(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)
            res = self.solve_2d(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)

        # self.layer_info_list = layer_info_list
        # self.T1 = T1

        # return de_ri_s, de_ri_p, de_ti_s, de_ti_p, R_s, R_p, T_s, T_p
        return res

    def conv_solve(self, **kwargs):
        # [setattr(self, k, v) for k, v in kwargs.items()]  # no need in npmeent

        if self.modeling_type_assigned == 0:  # Raster

            if self.fourier_type == 0:
                epx_conv_all, epy_conv_all, epz_conv_i_all = to_conv_mat_raster_discrete(
                    self.ucell, self.fto[0], self.fto[1], type_complex=self.type_complex,
                    enhanced_dfs=self.enhanced_dfs)

            elif self.fourier_type == 1:
                epx_conv_all, epy_conv_all, epz_conv_i_all = to_conv_mat_raster_continuous(
                    self.ucell, self.fto[0], self.fto[1], type_complex=self.type_complex)
            else:
                raise ValueError("Check 'modeling_type' and 'fourier_type' in 'conv_solve'.")

        elif self.modeling_type_assigned == 1:  # Vector
            ucell_vector = self.modeling_vector_instruction(self.ucell)
            self.grating_type_assigned = 2  # 1D conical
            epx_conv_all, epy_conv_all, epz_conv_i_all = to_conv_mat_vector(
                ucell_vector, self.fto[0], self.fto[1], type_complex=self.type_complex)

        else:
            raise ValueError("Check 'modeling_type' and 'fourier_type' in 'conv_solve'.")

        # de_ri_s, de_ri_p, de_ti_s, de_ti_p, layer_info_list, T1, R_s, R_p, T_s, T_p = self.solve(self.wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)
        res = self.solve(self.wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)

        # self.layer_info_list = layer_info_list
        # self.T1 = T1
        #
        # return de_ri_s, de_ri_p, de_ti_s, de_ti_p, R_s, R_p, T_s, T_p

        return res

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
        res = self.conv_solve()
        field_cell = self.calculate_field(res_x, res_y, res_z)
        return res, field_cell

    def field_plot(self, field_cell):
        field_plot(field_cell, self.pol)
