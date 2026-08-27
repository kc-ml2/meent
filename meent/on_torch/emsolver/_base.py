import torch

import numpy as np

from .scattering_method import scattering_1d_1, scattering_1d_2, scattering_1d_3, scattering_2d_1, scattering_2d_wv, \
    scattering_2d_2, scattering_2d_3

from .transfer_method import (transfer_1d_1, transfer_1d_2, transfer_1d_3, transfer_1d_4,
                              transfer_1d_conical_1, transfer_1d_conical_2, transfer_1d_conical_3, transfer_1d_conical_4,
                              transfer_2d_1, transfer_2d_2, transfer_2d_3, transfer_2d_4)


def _to_tensor(value, device, dtype):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    if isinstance(value, (list, tuple)) and any(isinstance(v, torch.Tensor) for v in value):
        return torch.stack([v.to(device=device, dtype=dtype).reshape(())
                            if isinstance(v, torch.Tensor)
                            else torch.tensor(v, device=device, dtype=dtype)
                            for v in value])
    return torch.tensor(value, device=device, dtype=dtype)


class _BaseRCWA:
    def __init__(self, n_top=1., n_bot=1., theta=0., phi=None, psi=None, pol=0., fto=(0, 0),
                 period=(1., 1.), wavelength=1.,
                 thickness=(0.,), connecting_algo='TMM', perturbation=1E-10,
                 device='cpu', type_complex=torch.complex128, use_pinv=False):

        if device in (0, 'cpu'):
            self._device = torch.device('cpu')
        elif device in (1, 'gpu', 'cuda'):
            self._device = torch.device('cuda')
        elif type(device) is torch.device:
            self._device = device
        else:
            raise ValueError('device')

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
        self.use_pinv = use_pinv

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
            self._theta = _to_tensor(theta, self.device, self.type_complex)
            self._theta = torch.where(self._theta == 0, self.perturbation, self._theta)

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, phi):
        if phi is None:
            self._phi = None
        else:
            self._phi = _to_tensor(phi, self.device, self.type_complex)

    @property
    def psi(self):
        return self._psi

    @psi.setter
    def psi(self, psi):
        if psi is not None:
            self._psi = _to_tensor(psi, self.device, self.type_complex)
            self._pol = -(2 * self._psi / torch.pi - 1)

    @property
    def pol(self):
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
            self._period = _to_tensor(period, self.device, self.type_float)
            if len(self._period) == 1:
                self._period = torch.cat([self._period, self._period])
        else:
            raise ValueError

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, thickness):
        if type(thickness) in (int, float):
            self._thickness = torch.tensor([thickness], device=self.device, dtype=self.type_float)
        elif type(thickness) in (list, tuple, np.ndarray) or isinstance(thickness, torch.Tensor):
            self._thickness = _to_tensor(thickness, self.device, self.type_float)
        else:
            raise ValueError

    def get_kx_ky_vector(self, wavelength):

        fto_x_range = torch.arange(-self.fto[0], self.fto[0] + 1, device=self.device,
                                   dtype=self.type_float)
        fto_y_range = torch.arange(-self.fto[1], self.fto[1] + 1, device=self.device,
                                   dtype=self.type_float)

        sin_theta = torch.sin(self.theta)

        if self.phi is None:
            phi = torch.tensor(0, device=self.device, dtype=self.type_complex)
        else:
            phi = self.phi

        kx = (self.n_top * sin_theta * torch.cos(phi) + fto_x_range * (
                wavelength / self.period[0])).type(self.type_complex).conj()

        ky = (self.n_top * sin_theta * torch.sin(phi) + fto_y_range * (
                wavelength / self.period[1])).type(self.type_complex).conj()

        # resolve_conj, not a bare conj: torch's conj() hands back a view carrying a lazy
        # conjugate bit, and numpy() refuses such a tensor rather than silently dropping the
        # conjugation. This is a public method, so callers reach for .numpy() on its result.
        return kx.resolve_conj(), ky.resolve_conj()

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
        else:
            raise ValueError

        for layer_index in range(len(self.thickness))[::-1]:

            epx_conv = epx_conv_all[layer_index]
            epy_conv = epy_conv_all[layer_index]
            epz_conv_i = epz_conv_i_all[layer_index]

            d = self.thickness[layer_index]

            if self.connecting_algo == 'TMM':
                W, V, q = transfer_1d_2(self.pol, kx, epx_conv, epy_conv, epz_conv_i, device=self.device,
                                        type_complex=self.type_complex, perturbation=self.perturbation,
                                        use_pinv=self.use_pinv)

                X, F, G, T, A_i, B = transfer_1d_3(k0, W, V, q, d, F, G, T, device=self.device,
                                                   type_complex=self.type_complex, use_pinv=self.use_pinv)

                layer_info = [epz_conv_i, W, V, q, d, A_i, B]
                self.layer_info_list.append(layer_info)

            elif self.connecting_algo == 'SMM':
                raise ValueError
            else:
                raise ValueError

        if self.connecting_algo == 'TMM':
            result, T1 = transfer_1d_4(self.pol, ff_x, F, G, T, kx, kz_top, kz_bot, self.theta, self.n_top, self.n_bot,
                                       device=self.device, type_complex=self.type_complex, use_pinv=self.use_pinv)
            self.T1 = T1

        elif self.connecting_algo == 'SMM':
            raise ValueError
        else:
            raise ValueError

        return result

    def solve_1d_conical(self, wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all):
        self.layer_info_list = []
        self.T1 = None

        ff_x = self.fto[0] * 2 + 1
        ff_y = 1

        k0 = 2 * torch.pi / wavelength
        kx, ky = self.get_kx_ky_vector(wavelength)

        if self.connecting_algo == 'TMM':
            kz_top, kz_bot, varphi, big_F, big_G, big_T \
                = transfer_1d_conical_1(kx, ky, self.n_top, self.n_bot, device=self.device,
                                        type_complex=self.type_complex)

        elif self.connecting_algo == 'SMM':
            print('SMM for 1D conical is not implemented')
            return np.nan, np.nan
        else:
            raise ValueError

        for layer_index in range(len(self.thickness))[::-1]:

            epx_conv = epx_conv_all[layer_index]
            epy_conv = epy_conv_all[layer_index]
            epz_conv_i = epz_conv_i_all[layer_index]

            d = self.thickness[layer_index]

            if self.connecting_algo == 'TMM':
                W, V, q = transfer_1d_conical_2(kx, ky, epx_conv, epy_conv, epz_conv_i, device=self.device,
                                                type_complex=self.type_complex, perturbation=self.perturbation,
                                                use_pinv=self.use_pinv)

                big_X, big_F, big_G, big_T, big_A_i, big_B, \
                    = transfer_1d_conical_3(k0, W, V, q, d, varphi, big_F, big_G, big_T, device=self.device,
                                            type_complex=self.type_complex, use_pinv=self.use_pinv)

                layer_info = [epz_conv_i, W, V, q, d, big_A_i, big_B]
                self.layer_info_list.append(layer_info)

            elif self.connecting_algo == 'SMM':
                raise ValueError
            else:
                raise ValueError

        if self.connecting_algo == 'TMM':
            result, big_T1 = transfer_1d_conical_4(kx,ff_x, ff_y, big_F, big_G, big_T, kz_top, kz_bot, self.psi,
                                                   self.theta, self.n_top, self.n_bot, device=self.device,
                                                   type_complex=self.type_complex,
                                                   use_pinv=self.use_pinv)
            self.T1 = big_T1

        elif self.connecting_algo == 'SMM':
            raise ValueError
        else:
            raise ValueError

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
        else:
            raise ValueError

        for layer_index in range(len(self.thickness))[::-1]:

            epx_conv = epx_conv_all[layer_index]
            epy_conv = epy_conv_all[layer_index]
            epz_conv_i = epz_conv_i_all[layer_index]

            d = self.thickness[layer_index]

            if self.connecting_algo == 'TMM':
                W, V, q = transfer_2d_2(kx, ky, epx_conv, epy_conv, epz_conv_i, device=self.device,
                                        type_complex=self.type_complex, perturbation=self.perturbation,
                                        use_pinv=self.use_pinv)

                big_X, big_F, big_G, big_T, big_A_i, big_B, \
                    = transfer_2d_3(k0, W, V, q, d, varphi, big_F, big_G, big_T, device=self.device,
                                    type_complex=self.type_complex, use_pinv=self.use_pinv)

                layer_info = [epz_conv_i, W, V, q, d, big_A_i, big_B]
                self.layer_info_list.append(layer_info)

            elif self.connecting_algo == 'SMM':
                raise ValueError
            else:
                raise ValueError

        if self.connecting_algo == 'TMM':

            result, big_T1 = transfer_2d_4(kx, ff_x, ff_y, big_F, big_G, big_T, kz_top, kz_bot, self.psi, self.theta,
                                           self.n_top, self.n_bot, device=self.device, type_complex=self.type_complex,
                                           use_pinv=self.use_pinv)
            self.T1 = big_T1

        elif self.connecting_algo == 'SMM':
            raise ValueError
        else:
            raise ValueError

        return result
