import torch

import numpy as np

from ._base import _BaseRCWA
from .convolution_matrix import to_conv_mat_raster_discrete, to_conv_mat_raster_continuous, to_conv_mat_vector
from .field_distribution import field_dist_1d, field_dist_1d_conical, field_dist_2d, field_plot


class ResultTorch:
    def __init__(self, res=None, res_te_inc=None, res_tm_inc=None):

        self.res = res
        self.res_te_inc = res_te_inc
        self.res_tm_inc = res_tm_inc

    @property
    def de_ri(self):
        if self.res is not None:
            return self.res.de_ri
        else:
            return None

    @property
    def de_ti(self):
        if self.res is not None:
            return self.res.de_ti
        else:
            return None


class ResultSubTorch:
    def __init__(self, R_s, R_p, T_s, T_p, de_ri, de_ri_s, de_ri_p, de_ti, de_ti_s, de_ti_p):
        self.R_s = R_s
        self.R_p = R_p
        self.T_s = T_s
        self.T_p = T_p
        self.de_ri = de_ri
        self.de_ri_s = de_ri_s
        self.de_ri_p = de_ri_p

        self.de_ti = de_ti
        self.de_ti_s = de_ti_s
        self.de_ti_p = de_ti_p


class RCWATorch(_BaseRCWA):
    def __init__(self,
                 n_top=1.,
                 n_bot=1.,
                 theta=0.,
                 phi=None,
                 psi=None,
                 period=(1., 1.),
                 wavelength=1.,
                 ucell=None,
                 thickness=(0.,),
                 backend=2,
                 pol=0.,
                 fto=(0, 0),
                 ucell_materials=None,
                 connecting_algo='TMM',
                 perturbation=1E-20,
                 device='cpu',
                 type_complex=torch.complex128,
                 fourier_type=0,
                 enhanced_dfs=True,
                 use_pinv=False,
                 ):

        super().__init__(n_top=n_top, n_bot=n_bot, theta=theta, phi=phi, psi=psi, pol=pol,
                         fto=fto, period=period, wavelength=wavelength,
                         thickness=thickness, connecting_algo=connecting_algo, perturbation=perturbation,
                         device=device, type_complex=type_complex, use_pinv=use_pinv)

        self._modeling_type_assigned = None
        self._grating_type_assigned = None

        self.ucell = ucell
        self.ucell_materials = ucell_materials

        self.backend = backend
        self.fourier_type = fourier_type
        self.enhanced_dfs = enhanced_dfs
        self.use_pinv = use_pinv

    @property
    def ucell(self):
        return self._ucell

    @ucell.setter
    def ucell(self, ucell):

        if isinstance(ucell, (torch.Tensor, np.ndarray)):  # Raster
            self._modeling_type_assigned = 0
            if ucell.dtype in (torch.complex128, torch.complex64):
                dtype = self.type_complex
                self._ucell = ucell.to(device=self.device, dtype=dtype)
            elif ucell.dtype in (torch.float64, torch.float32, torch.int64, torch.int32):
                dtype = self.type_float
                self._ucell = ucell.to(device=self.device, dtype=dtype)
            elif ucell.dtype in (np.int64, np.float64, np.int32, np.float32):
                dtype = self.type_float
                self._ucell = torch.tensor(ucell, device=self.device, dtype=dtype)
            elif ucell.dtype in (np.complex128, np.complex64):
                dtype = self.type_complex
                self._ucell = torch.tensor(ucell, device=self.device, dtype=dtype)
            else:
                raise ValueError

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
        # self.modeling_type_assigned = 0
        # self._grating_type_assigned = 1  # else

        if self.modeling_type_assigned == 0:
            if self.ucell.shape[1] == 1:
                if (self.pol in (0, 1)) and (self.phi is None) and (self.fto[1] == 0):
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

    def solve_for_conv(self, wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all):
        self._assign_grating_type()

        if self._grating_type_assigned == 0:
            result_dict = self.solve_1d(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)
        elif self._grating_type_assigned == 1:
            result_dict = self.solve_1d_conical(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)
        else:
            result_dict = self.solve_2d(wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)

        res_psi = ResultSubTorch(**result_dict['res']) if 'res' in result_dict else None
        res_te_inc = ResultSubTorch(**result_dict['res_te_inc']) if 'res_te_inc' in result_dict else None
        res_tm_inc = ResultSubTorch(**result_dict['res_tm_inc']) if 'res_tm_inc' in result_dict else None

        result = ResultTorch(res_psi, res_te_inc, res_tm_inc)

        return result

    def conv_solve(self, **kwargs):
        [setattr(self, k, v) for k, v in kwargs.items()]  # needed for optimization

        if self.modeling_type_assigned == 0:  # Raster

            if self.fourier_type == 0:
                epx_conv_all, epy_conv_all, epz_conv_i_all = to_conv_mat_raster_discrete(
                    self.ucell, self.fto[0], self.fto[1], device=self.device, type_complex=self.type_complex,
                    enhanced_dfs=self.enhanced_dfs, use_pinv=self.use_pinv)

            elif self.fourier_type == 1:
                epx_conv_all, epy_conv_all, epz_conv_i_all = to_conv_mat_raster_continuous(
                    self.ucell, self.fto[0], self.fto[1], device=self.device, type_complex=self.type_complex,
                    use_pinv=self.use_pinv)
            else:
                raise ValueError("Check 'modeling_type' and 'fourier_type' in 'conv_solve'.")

        elif self.modeling_type_assigned == 1:  # Vector
            ucell_vector = self.modeling_vector_instruction(self.ucell)
            epx_conv_all, epy_conv_all, epz_conv_i_all = to_conv_mat_vector(
                ucell_vector, self.fto[0], self.fto[1], device=self.device, type_complex=self.type_complex,
                use_pinv=self.use_pinv)

        else:
            raise ValueError("Check 'modeling_type' and 'fourier_type' in 'conv_solve'.")

        result = self.solve_for_conv(self.wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all)

        return result

    def calculate_field(self, res_x=20, res_y=20, res_z=20, set_field_input=(True, False, False)):
        kx, ky = self.get_kx_ky_vector(wavelength=self.wavelength)

        if self._grating_type_assigned == 0:
            res_y = 1
            field_cell = field_dist_1d(self.wavelength, kx, self.T1, self.layer_info_list, self.period, self.pol,
                                       res_x=res_x, res_y=res_y, res_z=res_z, device=self.device,
                                       type_complex=self.type_complex)
        elif self._grating_type_assigned == 1:
            field_cell = field_dist_1d_conical(self.wavelength, kx, ky, self.T1, self.layer_info_list, self.period,
                                               res_x=res_x, res_y=res_y, res_z=res_z, set_field_input=set_field_input,
                                               device=self.device, type_complex=self.type_complex)
        else:
            field_cell = field_dist_2d(self.wavelength, kx, ky, self.T1, self.layer_info_list, self.period,
                                       res_x=res_x, res_y=res_y, res_z=res_z, set_field_input=set_field_input,
                                       device=self.device, type_complex=self.type_complex)

        return field_cell

    def conv_solve_field(self, res_x=20, res_y=20, res_z=20, set_field_input=(True, False, False), **kwargs):
        [setattr(self, k, v) for k, v in kwargs.items()]  # needed for optimization

        res = self.conv_solve()
        field_cell = self.calculate_field(res_x, res_y, res_z, set_field_input)
        return res, field_cell

    def field_plot(self, field_cell):
        field_plot(field_cell, self.pol)
