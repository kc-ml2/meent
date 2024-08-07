import functools
import jax

import jax.numpy as jnp
import numpy as np

from .scattering_method import (scattering_1d_1, scattering_1d_2, scattering_1d_3,
                                scattering_2d_1, scattering_2d_wv, scattering_2d_2, scattering_2d_3)
from .transfer_method import (transfer_1d_1, transfer_1d_2, transfer_1d_3, transfer_1d_4,
                              transfer_2d_1, transfer_2d_2, transfer_2d_3, transfer_2d_4)


def jax_device_set(func):
    @functools.wraps(func)
    def wrap(*args, **kwargs):
        self, *_ = args
        with jax.default_device(self.device[0]):
            res = func(*args, **kwargs)
            return res

    return wrap


class _BaseRCWA:

    def __init__(self, n_top=1., n_bot=1., theta=0., phi=0., psi=None, pol=0., fto=(2, 0),
                 period=(100., 100.), wavelength=1.,
                 thickness=(0.,), connecting_algo='TMM', perturbation=1E-20,
                 device=0, type_complex=jnp.complex128):

        self.device = device

        # type_complex
        if type_complex in (0, jnp.complex128, np.complex128):
            self._type_complex = jnp.complex128
        elif type_complex in (1, jnp.complex64, np.complex64):
            self._type_complex = jnp.complex64
        else:
            raise ValueError('JAX type_complex')

        # currently these two are not used. Only TorchMeent uses.
        self._type_float = jnp.float64 if self._type_complex is not jnp.complex64 else jnp.float32
        self._type_int = jnp.int64 if self._type_complex is not jnp.complex64 else jnp.int32
        self.perturbation = perturbation

        self.n_top = n_top
        self.n_bot = n_bot

        # degree to radian due to JAX JIT
        self.theta = theta
        self.phi = phi
        self.pol = pol
        self.psi = psi
        # self._psi = jnp.array((jnp.pi / 2 * (1 - pol)), dtype=self.type_float)

        self.fto = fto
        self.period = period
        self.wavelength = wavelength
        self.thickness = thickness
        self.connecting_algo = connecting_algo
        self.layer_info_list = []
        self.T1 = None
        # self.kx = None  # only kx, not ky, because kx is always used while ky is 2D only.

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        if device in (0, 'cpu'):
            self._device = jax.devices('cpu')
        elif device in (1, 'gpu', 'cuda'):
            self._device = jax.devices('gpu')
        elif type(device) is list and (str(type(device[0])) == "<class 'jaxlib.xla_extension.Device'>"):
            self._device = device
        else:
            raise ValueError

    @property
    def type_complex(self):
        return self._type_complex

    @type_complex.setter
    def type_complex(self, type_complex):
        # type_complex
        if type_complex in (0, jnp.complex128, np.complex128):
            self._type_complex = jnp.complex128
        elif type_complex in (1, jnp.complex64, np.complex64):
            self._type_complex = jnp.complex64
        else:
            raise ValueError('JAX type_complex')

        self._type_float = jnp.float64 if self.type_complex is not jnp.complex64 else jnp.float32
        self._type_int = jnp.int64 if self.type_complex is not jnp.complex64 else jnp.int32
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
    def pol(self):
        return self._pol

    @pol.setter
    def pol(self, pol):
        room = 1E-6
        if 1 < pol < 1 + room:
            pol = 1
        elif 0 - room < pol < 0:
            pol = 0

        if not 0 <= pol <= 1:
            raise ValueError

        self._pol = pol
        psi = jnp.pi / 2 * (1 - self.pol)
        self._psi = jnp.array(psi, dtype=self.type_float)

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        self._theta = jnp.array(theta, dtype=self.type_float)
        self._theta = jnp.where(self._theta == 0, self.perturbation, self._theta)  # perturbation

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, phi):
        self._phi = jnp.array(phi, dtype=self.type_float)

    @property
    def psi(self):
        return self._psi

    @psi.setter
    def psi(self, psi):
        if psi is not None:
            self._psi = jnp.array(psi, dtype=self.type_float)
            pol = -(2 * psi / jnp.pi - 1)
            self._pol = pol

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
                raise ValueError
        elif isinstance(fto, np.ndarray) or isinstance(fto, jnp.ndarray):
            self._fto = fto.tolist()
            if type(self._fto) is list:
                if len(self._fto) == 1:
                    self._fto = [int(self._fto[0]), 0]
                elif len(self._fto) == 2:
                    self._fto = [int(v) for v in self._fto]
                else:
                    raise ValueError
            elif type(self._fto) in (int, float):
                self._fto = [int(self._fto), 0]
            else:
                raise ValueError
        elif type(fto) in (int, float):
            self._fto = [int(fto), 0]
        else:
            raise ValueError

    @property
    def period(self):
        return self._period

    @period.setter
    def period(self, period):
        if type(period) in (int, float):
            self._period = jnp.array([period, period], dtype=self.type_float)
        elif type(period) in (list, tuple, np.ndarray) or isinstance(period, jnp.ndarray):
            if len(period) == 1:
                period = [period[0], period[0]]
            self._period = jnp.array(period, dtype=self.type_float)
        elif type(period) is jax.interpreters.partial_eval.DynamicJaxprTracer:
            print('init period')
            jax.debug.print('init period')
            self._period = period
        else:
            raise ValueError

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, thickness):
        if type(thickness) in (int, float):
            self._thickness = jnp.array([thickness], dtype=self.type_float)
        elif type(thickness) in (list, tuple, np.ndarray):
            self._thickness = jnp.array(thickness, dtype=self.type_float)
        elif isinstance(thickness, jnp.ndarray):
            self._thickness = jnp.array(thickness, dtype=self.type_float)
        elif type(thickness) is jax.interpreters.partial_eval.DynamicJaxprTracer:
            print('init period')
            jax.debug.print('init period')
            self._thickness = thickness
        else:
            raise ValueError

    # @staticmethod
    # def jax_device_set(func):
    #     @functools.wraps(func)
    #     def wrap(*args, **kwargs):
    #         self, *_ = args
    #         with jax.default_device(self.device[0]):
    #             res = func(*args, **kwargs)
    #             return res
    #     return wrap

    @jax_device_set
    def get_kx_ky_vector(self, wavelength):

        fto_x_range = jnp.arange(-self.fto[0], self.fto[0] + 1)
        fto_y_range = jnp.arange(-self.fto[1], self.fto[1] + 1)

        kx_vector = (self.n_top * jnp.sin(self.theta) * jnp.cos(self.phi) + fto_x_range * (
                wavelength / self.period[0])).astype(self.type_complex)

        ky_vector = (self.n_top * jnp.sin(self.theta) * jnp.sin(self.phi) + fto_y_range * (
                wavelength / self.period[1])).astype(self.type_complex)

        return kx_vector, ky_vector

    @jax_device_set
    def solve_1d(self, wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all):
        self.layer_info_list = []
        self.T1 = None

        ff_x = self.fto[0] * 2 + 1

        k0 = 2 * jnp.pi / wavelength
        kx, _ = self.get_kx_ky_vector(wavelength)

        if self.connecting_algo == 'TMM':
            kz_top, kz_bot, F, G, T \
                = transfer_1d_1(self.pol, ff_x, kx, self.n_top, self.n_bot, type_complex=self.type_complex)
        elif self.connecting_algo == 'SMM':
            Kx, Wg, Vg, Kzg, Wr, Vr, Kzr, Wt, Vt, Kzt, Ar, Br, Sg \
                = scattering_1d_1(k0, self.n_top, self.n_bot, self.theta, self.phi, self.period,
                                  self.pol, wl=wavelength)
        else:
            raise ValueError

        # From the last layer
        for layer_index in range(len(self.thickness))[::-1]:

            epx_conv = epx_conv_all[layer_index]
            epy_conv = epy_conv_all[layer_index]
            epz_conv_i = epz_conv_i_all[layer_index]

            d = self.thickness[layer_index]

            if self.connecting_algo == 'TMM':
                W, V, q = transfer_1d_2(self.pol, kx, epx_conv, epy_conv, epz_conv_i, self.type_complex)

                X, F, G, T, A_i, B = transfer_1d_3(k0, W, V, q, d, F, G, T, type_complex=self.type_complex)

                layer_info = [epz_conv_i, W, V, q, d, A_i, B]
                self.layer_info_list.append(layer_info)

            elif self.connecting_algo == 'SMM':
                A, B, S_dict, Sg = scattering_1d_2(W, Wg, V, Vg, d, k0, Q, Sg)
            else:
                raise ValueError

        if self.connecting_algo == 'TMM':
            de_ri, de_ti, T1 = transfer_1d_4(self.pol, F, G, T, kz_top, kz_bot, self.theta, self.n_top, self.n_bot,
                                             type_complex=self.type_complex)
            self.T1 = T1

        elif self.connecting_algo == 'SMM':
            de_ri, de_ti = scattering_1d_3(Wt, Wg, Vt, Vg, Sg, ff, Wr, self.fto, Kzr, Kzt,
                                           self.n_top, self.n_bot, self.theta, self.pol)
        else:
            raise ValueError

        return de_ri, de_ti, self.layer_info_list, self.T1
    # @jax_device_set
    # def solve_1d_conical(self, wavelength, E_conv_all, o_E_conv_all):
    #
    #     self.layer_info_list = []
    #     self.T1 = None
    #
    #     # fourier_indices = jnp.arange(-self.fto, self.fto + 1)
    #     ff = self.fto[0] * 2 + 1
    #
    #     delta_i0 = jnp.zeros(ff, dtype=self.type_complex)
    #     delta_i0 = delta_i0.at[self.fto[0]].set(1)
    #
    #     k0 = 2 * jnp.pi / wavelength
    #
    #     if self.connecting_algo == 'TMM':
    #         Kx, ky, k_I_z, k_II_z, varphi, Y_I, Y_II, Z_I, Z_II, big_F, big_G, big_T \
    #             = transfer_1d_conical_1(ff, k0, self.n_top, self.n_bot, self.kx, self.theta, self.phi,
    #                                     type_complex=self.type_complex)
    #     elif self.connecting_algo == 'SMM':
    #         print('SMM for 1D conical is not implemented')
    #         return jnp.nan, jnp.nan
    #     else:
    #         raise ValueError
    #
    #     # for E_conv, o_E_conv, d in zip(E_conv_all[::-1], o_E_conv_all[::-1], self.thickness[::-1]):
    #     count = min(len(E_conv_all), len(o_E_conv_all), len(self.thickness))
    #
    #     # From the last layer
    #     for layer_index in range(count)[::-1]:
    #
    #         E_conv = E_conv_all[layer_index]
    #         # o_E_conv = o_E_conv_all[layer_index]
    #         o_E_conv = None
    #
    #         d = self.thickness[layer_index]
    #
    #         E_conv_i = jnp.linalg.inv(E_conv)
    #         # o_E_conv_i = jnp.linalg.inv(o_E_conv)
    #         o_E_conv_i = None
    #
    #         if self.connecting_algo == 'TMM':
    #             big_X, big_F, big_G, big_T, big_A_i, big_B, W_1, W_2, V_11, V_12, V_21, V_22, q_1, q_2 \
    #                 = transfer_1d_conical_2(k0, Kx, ky, E_conv, E_conv_i, o_E_conv_i, ff, d,
    #                                         varphi, big_F, big_G, big_T,
    #                                         type_complex=self.type_complex, device=self.device)
    #
    #             layer_info = [E_conv_i, q_1, q_2, W_1, W_2, V_11, V_12, V_21, V_22, big_X, big_A_i, big_B, d]
    #             self.layer_info_list.append(layer_info)
    #
    #         elif self.connecting_algo == 'SMM':
    #             raise ValueError
    #         else:
    #             raise ValueError
    #
    #     if self.connecting_algo == 'TMM':
    #         de_ri, de_ti, big_T1 = transfer_1d_conical_3(big_F, big_G, big_T, Z_I, Y_I, self.psi, self.theta, ff,
    #                                                      delta_i0, k_I_z, k0, self.n_top, self.n_bot, k_II_z,
    #                                                      type_complex=self.type_complex)
    #         self.T1 = big_T1
    #
    #     elif self.connecting_algo == 'SMM':
    #         raise ValueError
    #     else:
    #         raise ValueError
    #
    #     return de_ri, de_ti, self.layer_info_list, self.T1

    @jax_device_set
    def solve_2d(self, wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all):

        self.layer_info_list = []
        self.T1 = None

        ff_x = self.fto[0] * 2 + 1
        ff_y = self.fto[1] * 2 + 1

        k0 = 2 * jnp.pi / wavelength
        kx, ky = self.get_kx_ky_vector(wavelength)

        if self.connecting_algo == 'TMM':
            kz_top, kz_bot, varphi, big_F, big_G, big_T \
                = transfer_2d_1(ff_x, ff_y, kx, ky, self.n_top, self.n_bot, type_complex=self.type_complex)

        elif self.connecting_algo == 'SMM':
            Kx, Ky, kz_inc, Wg, Vg, Kzg, Wr, Vr, Kzr, Wt, Vt, Kzt, Ar, Br, Sg \
                = scattering_2d_1(self.n_top, self.n_bot, self.theta, self.phi, k0, self.period, self.fto)
        else:
            raise ValueError

        # From the last layer
        for layer_index in range(len(self.thickness))[::-1]:

            epx_conv = epx_conv_all[layer_index]
            epy_conv = epy_conv_all[layer_index]
            epz_conv_i = epz_conv_i_all[layer_index]

            d = self.thickness[layer_index]

            if self.connecting_algo == 'TMM':
                W, V, q = transfer_2d_2(kx, ky, epx_conv, epy_conv, epz_conv_i, type_complex=self.type_complex)

                big_X, big_F, big_G, big_T, big_A_i, big_B, \
                    = transfer_2d_3(k0, W, V, q, d, varphi, big_F, big_G, big_T, type_complex=self.type_complex)

                layer_info = [epz_conv_i, W, V, q, d, big_A_i, big_B]
                self.layer_info_list.append(layer_info)

            elif self.connecting_algo == 'SMM':
                W, V, q = scattering_2d_wv(ff_xy, Kx, Ky, E_conv, o_E_conv, o_E_conv_i, E_conv_i)
                A, B, Sl_dict, Sg_matrix, Sg = scattering_2d_2(W, Wg, V, Vg, d, k0, Sg, q)
            else:
                raise ValueError

        if self.connecting_algo == 'TMM':
            de_ri, de_ti, big_T1 = transfer_2d_4(big_F, big_G, big_T, kz_top, kz_bot, self.psi, self.theta,
                                                 self.n_top, self.n_bot, type_complex=self.type_complex)
            self.T1 = big_T1

        elif self.connecting_algo == 'SMM':
            de_ri, de_ti = scattering_2d_3(ff_xy, Wt, Wg, Vt, Vg, Sg, Wr, Kx, Ky, Kzr, Kzt, kz_inc, self.n_top,
                                           self.pol, self.theta, self.phi, self.fto)
        else:
            raise ValueError
        de_ri = de_ri.reshape((ff_y, ff_x)).T
        de_ti = de_ti.reshape((ff_y, ff_x)).T

        return de_ri, de_ti, self.layer_info_list, self.T1

