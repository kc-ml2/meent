import torch

import numpy as np

from ._base import _BaseRCWA
from .convolution_matrix import to_conv_mat_raster_discrete, to_conv_mat_raster_continuous, to_conv_mat_vector
from .field_distribution import field_dist_1d, field_dist_1d_conical, field_dist_2d, field_plot


class ResultTorch:
    """What a solve returns: up to three sub-results, one per incident polarization.

    The general path solves TE and TM incidence anyway - the boundary condition is built for
    both and solved in one linear system - so both are handed back rather than discarded.
    `res` is the combination the user actually asked for via `psi`. On the fast 1D TE/TM path
    only `res` exists, since that path solves one polarization by construction.

    de_ri / de_ti are forwarded from `res` so the common case reads off this object directly.
    """

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
    """One incidence's answer, per diffraction order.

    R / T are complex amplitudes and keep sign and phase; de_ri / de_ti are the efficiencies,
    |amplitude|^2 weighted by the order's real kz. The efficiencies are what sums to 1 on a
    lossless structure, and the amplitudes are what a convention error shows up in - squaring
    discards exactly the sign and phase such an error lives in.
    """

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
                 perturbation=1E-10,
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

        self.is_aniso = False
        self._modeling_type_assigned = None
        self._grating_type_assigned = None

        self.ucell = ucell
        self.ucell_materials = ucell_materials
        self._assign_grating_type()

        self.backend = backend
        self.fourier_type = fourier_type
        self.enhanced_dfs = enhanced_dfs
        self.use_pinv = use_pinv

    @property
    def ucell(self):
        return self._ucell

    @ucell.setter
    def ucell(self, ucell):
        """Accepts either modeling style, and the type of what is assigned decides which.

        An array is a raster (modeling_type 0); a list is the vector modeler's layer
        description (modeling_type 1). Nothing else selects between them - there is no flag.
        """
        if isinstance(ucell, (torch.Tensor, np.ndarray)):
            self._modeling_type_assigned = 0

            if isinstance(ucell, np.ndarray):
                ucell = torch.from_numpy(ucell)

            if ucell.dim() == 3:
                pass
            elif ucell.dim() == 4:
                if ucell.shape[-1] != 3:
                    raise ValueError("Anisotropic ucell must have 3 components (nx, ny, nz) in the last dimension")
            else:
                raise ValueError("ucell must be 3D (Isotropic) or 4D (Anisotropic)")

            # A real ucell stays real: a lossless structure costs half the memory and does not
            # need the imaginary half. It is promoted to complex later, where it has to be.
            if ucell.dtype in (torch.complex128, torch.complex64):
                dtype = self.type_complex
            else:
                dtype = self.type_float

            self._ucell = ucell.to(device=self.device, dtype=dtype)
        elif type(ucell) is list:
            self._modeling_type_assigned = 1
            self._ucell = ucell
        elif ucell is None:
            self._ucell = ucell
        else:
            raise ValueError("Invalid ucell type. Expected Tensor, ndarray, or list")

    @property
    def modeling_type_assigned(self):
        return self._modeling_type_assigned


    def _assign_grating_type(self):
        """Route the problem to the cheapest solver that can still represent it.

            0  scalar 1D    TE and TM decouple entirely; one scalar system
            1  1D conical   1D structure, but the incidence is out of plane, so they couple
            2  2D           the general case

        Speed is the only motive - 2 is correct for every case here. Note this is re-derived
        at each solve rather than cached: `ucell`, `phi` and `pol` can all be reassigned after
        construction, which every example does, and any of them can change the answer.
        """
        if self.modeling_type_assigned == 0:
            # Anisotropy that is only nominal - a 4D ucell whose three components agree
            # everywhere - is treated as isotropic, so it keeps access to the cheaper routes.
            if self.ucell.dim() == 4:
                nx, ny, nz = self.ucell[..., 0], self.ucell[..., 1], self.ucell[..., 2]
                self.is_aniso = not (torch.allclose(nx, ny) and torch.allclose(ny, nz))
            else:
                self.is_aniso = False

            if self.ucell.shape[1] == 1:  # one row in y: the structure is 1D
                # phi=0 and phi=None are the same physical problem, and both take the scalar
                # path. The numpy and jax backends differ here - there phi=0 forces conical.
                phi_is_zero_or_none = self.phi is None or self.phi == 0
                if (self.pol in (0, 1)) and phi_is_zero_or_none and (self.fto[1] == 0):
                    self._grating_type_assigned = 0
                elif self.is_aniso:
                    # Off-diagonal coupling breaks the conical form's assumptions, so a
                    # genuinely anisotropic 1D layer still goes the 2D route.
                    self._grating_type_assigned = 2
                else:
                    self._grating_type_assigned = 1
            else:
                self._grating_type_assigned = 2
        elif self.modeling_type_assigned == 1:
            # The vector modeler always produces a 2D description, even for a 1D structure.
            self._grating_type_assigned = 2

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
        """Build the convolution matrices for the current ucell, then solve. The main entry point.

        kwargs are assigned onto self first, so a sweep or an optimization step can change one
        parameter and re-solve in a single call.
        """
        [setattr(self, k, v) for k, v in kwargs.items()]

        if self.modeling_type_assigned == 0:

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

        elif self.modeling_type_assigned == 1:
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

        # The scalar 1D path solves only the three components that are non-zero for its
        # polarization; every caller downstream expects all six in (Ex, Ey, Ez, Hx, Hy, Hz)
        # order. Widen it here, with explicit zeros in the slots that path cannot populate, so
        # the return shape does not depend on which solver ran. TE carries (Ey, Hx, Hz) and TM
        # carries (Hy, Ex, Ez) - hence the two different scatter patterns.
        if field_cell.shape[-1] == 3:
            zero = torch.zeros_like(field_cell[..., 0])
            if self.pol == 0:
                field_cell = torch.stack((zero, field_cell[..., 0], zero,
                                          field_cell[..., 1], zero, field_cell[..., 2]), dim=-1)
            else:
                field_cell = torch.stack((field_cell[..., 1], zero, field_cell[..., 2],
                                          zero, field_cell[..., 0], zero), dim=-1)

        # Time convention only: meent solves on exp(+j*w*t), RETICOLO reports on exp(-i*w*t),
        # and the same real field is Re[E e^{jwt}] = Re[E* e^{-iwt}]. No component signs are
        # applied on top. A (-1, 1, -1, 1, -1, 1) sign vector used to sit here together with
        # a global negation of the scalar-1D TM field; the two cancelled exactly on that path
        # (it has only Ex, Ez, Hy, which the vector negates) and so were invisible there,
        # while on the conical and 2D paths the vector alone was left over and put every
        # field off by exactly that pattern. The p-basis flip that belongs to it now lives
        # where it is a convention rather than a correction - P_TM_SIGN in transfer_method.
        # z increases upward, from the substrate side toward the superstrate, so the stored
        # axis runs the way the physical z axis does. The solver assembles the layers
        # top-down because that is the order it sweeps them in; that is a loop order, not a
        # coordinate. Flipping here also reverses the samples inside each layer, which is
        # what makes the whole axis monotonic rather than sawtoothed.
        field_cell = field_cell.flip(-4)

        # resolve_conj, not a bare conj: torch's conj() returns a view carrying a lazy
        # conjugate bit, and numpy() refuses such a tensor outright rather than silently
        # dropping the conjugation. A public return value has to be an ordinary tensor.
        return field_cell.conj().resolve_conj()

    def conv_solve_field(self, res_x=20, res_y=20, res_z=20,
                         set_field_input=(True, False, False), **kwargs):
        [setattr(self, k, v) for k, v in kwargs.items()]

        res = self.conv_solve()
        field_cell = self.calculate_field(res_x, res_y, res_z, set_field_input)
        return res, field_cell

    def field_plot(self, field_cell):
        field_plot(field_cell, self.pol)
