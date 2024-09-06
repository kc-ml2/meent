import torch

import numpy as np

from .scattering_method import scattering_1d_1, scattering_1d_2, scattering_1d_3, scattering_2d_1, scattering_2d_wv, \
    scattering_2d_2, scattering_2d_3

from .transfer_method import (transfer_1d_1, transfer_1d_2, transfer_1d_3, transfer_1d_4,
                              transfer_2d_1, transfer_2d_2, transfer_2d_3, transfer_2d_4)


class _BaseRCWA:
    def __init__(self, n_top=1., n_bot=1., theta=0., phi=None, psi=None, pol=0., fto=(0, 0),
                 period=(1., 1.), wavelength=1.,
                 thickness=(0.,), connecting_algo='TMM', perturbation=1E-20,
                 device='cpu', type_complex=torch.complex128):

        # device
        if device in (0, 'cpu'):
            self._device = torch.device('cpu')
        elif device in (1, 'gpu', 'cuda'):
            self._device = torch.device('cuda')
        elif type(device) is torch.device:
            self._device = device
        else:
            raise ValueError('device')

        # type_complex
        if type_complex in (0, torch.complex128, np.complex128):
            self._type_complex = torch.complex128
        elif type_complex in (1, torch.complex64, np.complex64):
            self._type_complex = torch.complex64
        else:
            raise ValueError('Torch type_complex')

        self._type_float = torch.float64 if self._type_complex is not torch.complex64 else torch.float32
        self._type_int = torch.int64 if self._type_complex is not torch.complex64 else torch.int32
        self.perturbation = perturbation

        self.n_top = n_top
        self.n_bot = n_bot

        self.theta = theta
        self.phi = phi
        self.pol = pol
        self.psi = psi

        self.fto = fto
        self.period = period
        self.wavelength = wavelength
        self.thickness = thickness
        self.connecting_algo = connecting_algo
        self.layer_info_list = []
        self.T1 = None

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        if device == 0:
            self._device = torch.device('cpu')
        elif device == 1:
            self._device = torch.device('cuda')
        elif type(device) is torch.device:
            self._device = device
        else:
            raise ValueError

    @property
    def type_complex(self):
        return self._type_complex

    @type_complex.setter
    def type_complex(self, type_complex):
        if type_complex in (0, torch.complex128, np.complex128):
            self._type_complex = torch.complex128
        elif type_complex in (1, torch.complex64, np.complex64):
            self._type_complex = torch.complex64
        else:
            raise ValueError('type_complex')

        self._type_float = torch.float64 if self.type_complex is not torch.complex64 else torch.float32
        self._type_int = torch.int64 if self.type_complex is not torch.complex64 else torch.int32
        self.theta = self.theta
        self.phi = self.phi
        self._psi = self.psi

        self.fto = self.fto
        self.thickness = self.thickness

    @property
    def type_float(self):
        return self._type_float

    @property
    def type_int(self):
        return self._type_int

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        if theta is None:
            self._theta = None
        else:
            self._theta = torch.tensor(theta, device=self.device, dtype=self.type_complex)
            self._theta = torch.where(self._theta == 0, self.perturbation, self._theta)  # perturbation

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, phi):
        if phi is None:
            self._phi = None
        else:
            self._phi = torch.tensor(phi, device=self.device, dtype=self.type_complex)

    @property
    def psi(self):
        return self._psi

    @psi.setter
    def psi(self, psi):
        if psi is not None:
            self._psi = torch.tensor(psi, dtype=self.type_complex)
            pol = -(2 * psi / torch.pi - 1)
            self._pol = pol

    @property
    def pol(self):
        """
        portion of TM. 0: full TE, 1: full TM

        Returns: polarization ratio

        """
        return self._pol

    @pol.setter
    def pol(self, pol):
        if not 0 <= pol <= 1:
            raise ValueError

        self._pol = pol
        psi = torch.tensor(torch.pi / 2 * (1 - self.pol), device=self.device, dtype=self.type_complex)
        self._psi = psi

    @property
    def fto(self):
        return self._fto

    @fto.setter
    def fto(self, fto):

        if type(fto) in (list, tuple):
            if len(fto) == 1:
                self._fto = [int(fto[0]), 0]
            elif len(fto) == 2:
                self._fto = [int(v) for v in fto]
            else:
                raise ValueError('Torch fto')
        elif isinstance(fto, np.ndarray) or isinstance(fto, torch.Tensor):
            self._fto = fto.tolist()
            if type(self._fto) is list:
                if len(self._fto) == 1:
                    self._fto = [int(self._fto[0]), 0]
                elif len(self._fto) == 2:
                    self._fto = [int(v) for v in self._fto]
                else:
                    raise ValueError('Torch fto')
            elif type(self._fto) in (int, float):
                self._fto = [int(self._fto), 0]
            else:
                raise ValueError('Torch fto')
        elif type(fto) in (int, float):
            self._fto = [int(fto), 0]
        else:
            raise ValueError('Torch fto')

    @property
    def period(self):
        return self._period

    @period.setter
    def period(self, period):
        if type(period) in (int, float):
            self._period = torch.tensor([period, period], device=self.device, dtype=self.type_float)
        elif type(period) in (list, tuple, np.ndarray) or isinstance(period, torch.Tensor):
            if len(period) == 1:
                period = [period[0], period[0]]
            self._period = torch.tensor(period, device=self.device, dtype=self.type_float)
        else:
            raise ValueError

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, thickness):
        if type(thickness) in (int, float):
            self._thickness = torch.tensor([thickness], device=self.device, dtype=self.type_float)
        elif type(thickness) in (list, tuple, np.ndarray):
            self._thickness = torch.tensor(thickness, device=self.device, dtype=self.type_float)
        elif type(thickness) is torch.Tensor:
            self._thickness = thickness.to(device=self.device, dtype=self.type_float)
        else:
            raise ValueError

    def get_kx_ky_vector(self, wavelength):

        fto_x_range = torch.arange(-self.fto[0], self.fto[0] + 1, device=self.device,
                                   dtype=self.type_float)
        fto_y_range = torch.arange(-self.fto[1], self.fto[1] + 1, device=self.device,
                                   dtype=self.type_float)

        if self.theta.real >= torch.tensor(torch.pi/2, dtype=torch.float32):
            # https://github.com/numpy/numpy/issues/27306
            sin_theta = torch.sin(
                torch.nextafter(torch.float32(torch.pi / 2), torch.float32(0)) + self.theta.imag * np.complex64(1j))
        else:
            sin_theta = np.sin(self.theta)

        if self.phi is None:
            phi = torch.tensor(0, device=self.device, dtype=self.type_complex)
        else:
            phi = self.phi

        kx = (self.n_top * sin_theta * torch.cos(phi) + fto_x_range * (
                wavelength / self.period[0])).type(self.type_complex).conj()

        ky = (self.n_top * sin_theta * torch.sin(phi) + fto_y_range * (
                wavelength / self.period[1])).type(self.type_complex).conj()

        return kx, ky

    def solve_1d(self, wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all):
        self.layer_info_list = []
        self.T1 = None

        ff_x = self.fto[0] * 2 + 1

        k0 = 2 * torch.pi / wavelength
        kx, _ = self.get_kx_ky_vector(wavelength)

        if self.connecting_algo == 'TMM':
            kz_top, kz_bot, F, G, T \
                = transfer_1d_1(self.pol, kx, self.n_top, self.n_bot, device=self.device,
                                type_complex=self.type_complex)

        elif self.connecting_algo == 'SMM':
            raise ValueError
            # Kx, Wg, Vg, Kzg, Wr, Vr, Kzr, Wt, Vt, Kzt, Ar, Br, Sg \
            #     = scattering_1d_1(k0, self.n_top, self.n_bot, self.theta, self.phi, fourier_indices, self.period,
            #                       self.pol, wl=wavelength)
        else:
            raise ValueError

        # From the last layer
        for layer_index in range(len(self.thickness))[::-1]:

            epx_conv = epx_conv_all[layer_index]
            epy_conv = epy_conv_all[layer_index]
            epz_conv_i = epz_conv_i_all[layer_index]

            d = self.thickness[layer_index]

            if self.connecting_algo == 'TMM':
                W, V, q = transfer_1d_2(self.pol, kx, epx_conv, epy_conv, epz_conv_i, device=self.device,
                                        type_complex=self.type_complex)

                X, F, G, T, A_i, B = transfer_1d_3(k0, W, V, q, d, F, G, T, device=self.device,
                                                   type_complex=self.type_complex)

                layer_info = [epz_conv_i, W, V, q, d, A_i, B]
                self.layer_info_list.append(layer_info)

            elif self.connecting_algo == 'SMM':
                raise ValueError
                # A, B, S_dict, Sg = scattering_1d_2(W, Wg, V, Vg, d, k0, Q, Sg)
            else:
                raise ValueError

        if self.connecting_algo == 'TMM':
            result, T1 = transfer_1d_4(self.pol, ff_x, F, G, T, kz_top, kz_bot, self.theta, self.n_top, self.n_bot,
                                       device=self.device, type_complex=self.type_complex)
            self.T1 = T1

        elif self.connecting_algo == 'SMM':
            raise ValueError
            # de_ri, de_ti = scattering_1d_3(Wt, Wg, Vt, Vg, Sg, self.ff, Wr, self.fto, Kzr, Kzt,
            #                                self.n_top, self.n_bot, self.theta, self.pol)
        else:
            raise ValueError

        # return de_ri, de_ti, self.rayleigh_R, self.rayleigh_T, self.layer_info_list, self.T1
        return result

    def solve_2d(self, wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all):

        self.layer_info_list = []
        self.T1 = None

        ff_x = self.fto[0] * 2 + 1
        ff_y = self.fto[1] * 2 + 1

        k0 = 2 * torch.pi / wavelength
        kx, ky = self.get_kx_ky_vector(wavelength)

        if self.connecting_algo == 'TMM':
            kz_top, kz_bot, varphi, big_F, big_G, big_T \
                = transfer_2d_1(kx, ky, self.n_top, self.n_bot, device=self.device, type_complex=self.type_complex)

        elif self.connecting_algo == 'SMM':
            raise ValueError
            # Kx, Ky, kz_inc, Wg, Vg, Kzg, Wr, Vr, Kzr, Wt, Vt, Kzt, Ar, Br, Sg \
            #     = scattering_2d_1(self.n_top, self.n_bot, self.theta, self.phi, k0, self.period, self.fto)
        else:
            raise ValueError

        # From the last layer
        for layer_index in range(len(self.thickness))[::-1]:

            epx_conv = epx_conv_all[layer_index]
            epy_conv = epy_conv_all[layer_index]
            epz_conv_i = epz_conv_i_all[layer_index]

            d = self.thickness[layer_index]

            if self.connecting_algo == 'TMM':
                W, V, q = transfer_2d_2(kx, ky, epx_conv, epy_conv, epz_conv_i, device=self.device,
                                        type_complex=self.type_complex)

                big_X, big_F, big_G, big_T, big_A_i, big_B, \
                    = transfer_2d_3(k0, W, V, q, d, varphi, big_F, big_G, big_T, device=self.device,
                                    type_complex=self.type_complex)

                layer_info = [epz_conv_i, W, V, q, d, big_A_i, big_B]
                self.layer_info_list.append(layer_info)

            elif self.connecting_algo == 'SMM':
                raise ValueError
                # W, V, LAMBDA = scattering_2d_wv(ff_xy, Kx, Ky, E_conv, o_E_conv, o_E_conv_i, E_conv_i)
                # A, B, Sl_dict, Sg_matrix, Sg = scattering_2d_2(W, Wg, V, Vg, d, k0, Sg, LAMBDA)
            else:
                raise ValueError

        if self.connecting_algo == 'TMM':
            # de_ri, de_ti, big_T1, [R_s, R_p], [T_s, T_p], = transfer_2d_4(big_F, big_G, big_T, kz_top, kz_bot, self.psi, self.theta,
            #                                      self.n_top, self.n_bot, device=self.device,
            #                                      type_complex=self.type_complex)
            # self.T1 = big_T1
            # self.rayleigh_R = [R_s, R_p]
            # self.rayleigh_T = [T_s, T_p]

            # de_ri_s, de_ri_p, de_ti_s, de_ti_p, big_T1, R_s, R_p, T_s, T_p = transfer_2d_4(big_F, big_G, big_T, kz_top,
            #                                                                                kz_bot, self.psi, self.theta,
            #                                                                                self.n_top, self.n_bot,
            #                                                                                type_complex=self.type_complex)
            result, big_T1 = transfer_2d_4(ff_x, ff_y, big_F, big_G, big_T, kz_top, kz_bot, self.psi, self.theta,
                                           self.n_top, self.n_bot, type_complex=self.type_complex)
            self.T1 = big_T1

        elif self.connecting_algo == 'SMM':
            raise ValueError
            # de_ri, de_ti = scattering_2d_3(Wt, Wg, Vt, Vg, Sg, Wr, Kx, Ky, Kzr, Kzt, kz_inc, self.n_top,
            #                                self.pol, self.theta, self.phi, self.fto)
        else:
            raise ValueError

        # de_ri = de_ri.reshape((ff_y, ff_x)).T
        # de_ti = de_ti.reshape((ff_y, ff_x)).T
        #
        # return de_ri, de_ti, self.rayleigh_R, self.rayleigh_T, self.layer_info_list, self.T1
        # return de_ri_s, de_ri_p, de_ti_s, de_ti_p, self.layer_info_list, self.T1, R_s, R_p, T_s, T_p
        return result
