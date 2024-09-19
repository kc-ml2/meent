import functools
import jax

import jax.numpy as jnp
import numpy as np

from .scattering_method import (scattering_1d_1, scattering_1d_2, scattering_1d_3,
                                scattering_2d_1, scattering_2d_wv, scattering_2d_2, scattering_2d_3)
from .transfer_method import (transfer_1d_1, transfer_1d_2, transfer_1d_3, transfer_1d_4, transfer_1d_conical_1,
                              transfer_1d_conical_2, transfer_1d_conical_3, transfer_1d_conical_4,
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

    def __init__(self, n_top=1., n_bot=1., theta=0., phi=None, psi=None, pol=0., fto=(0, 0),
                 period=(1., 1.), wavelength=1.,
                 thickness=(0.,), connecting_algo='TMM', perturbation=1E-20,
                 device=0, type_complex=jnp.complex128, use_pinv=False):

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
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        if theta is None:
            self._theta = None
        else:
            self._theta = jnp.array(theta, dtype=self.type_complex)
            self._theta = jnp.where(self._theta == 0, self.perturbation, self._theta)  # perturbation

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, phi):
        if phi is None:
            self._phi = None
        else:
            self._phi = jnp.array(phi, dtype=self.type_complex)

    @property
    def psi(self):
        return self._psi

    @psi.setter
    def psi(self, psi):
        if psi is not None:
            self._psi = jnp.array(psi, dtype=self.type_complex)
            pol = -(2 * psi / jnp.pi - 1)
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
        room = 1E-6
        if 1 < pol < 1 + room:
            pol = 1
        elif 0 - room < pol < 0:
            pol = 0

        if not 0 <= pol <= 1:
            raise ValueError

        self._pol = pol
        psi = jnp.array(jnp.pi / 2 * (1 - self.pol), dtype=self.type_complex)
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

        def adjust_theta():
            # https://github.com/numpy/numpy/issues/27306
            check = self.theta.real >= jnp.float32(jnp.pi / 2)
            sin_theta_true_case = jnp.sin(
                jnp.nextafter(jnp.float32(jnp.pi / 2), jnp.float32(0)) + self.theta.imag * jnp.complex64(1j))
            sin_theta_false_case = jnp.sin(self.theta)
            return jnp.where(check, sin_theta_true_case, sin_theta_false_case)

        sin_theta = adjust_theta()

        phi = 0 if self.phi is None else self.phi  # phi is None -> 1D TE TM case

        kx = (self.n_top * sin_theta * jnp.cos(phi) + fto_x_range * (
                wavelength / self.period[0])).astype(self.type_complex).conj()

        ky = (self.n_top * sin_theta * jnp.sin(phi) + fto_y_range * (
                wavelength / self.period[1])).astype(self.type_complex).conj()

        return kx, ky

    @jax_device_set
    # @jax.jit  # TODO: make optional
    def solve_1d(self, wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all):
        self.layer_info_list = []
        self.T1 = None

        ff_x = self.fto[0] * 2 + 1

        k0 = 2 * jnp.pi / wavelength
        kx, _ = self.get_kx_ky_vector(wavelength)

        if self.connecting_algo == 'TMM':
            kz_top, kz_bot, F, G, T \
                = transfer_1d_1(self.pol, kx, self.n_top, self.n_bot, type_complex=self.type_complex)
        elif self.connecting_algo == 'SMM':
            raise ValueError

            # Kx, Wg, Vg, Kzg, Wr, Vr, Kzr, Wt, Vt, Kzt, Ar, Br, Sg \
            #     = scattering_1d_1(k0, self.n_top, self.n_bot, self.theta, self.phi, self.period,
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
                W, V, q = transfer_1d_2(self.pol, kx, epx_conv, epy_conv, epz_conv_i, self.type_complex,
                                        self.perturbation, use_pinv=self.use_pinv)

                X, F, G, T, A_i, B = transfer_1d_3(k0, W, V, q, d, F, G, T, type_complex=self.type_complex,
                                                   use_pinv=self.use_pinv)

                layer_info = [epz_conv_i, W, V, q, d, A_i, B]
                self.layer_info_list.append(layer_info)

            elif self.connecting_algo == 'SMM':
                raise ValueError
                # A, B, S_dict, Sg = scattering_1d_2(W, Wg, V, Vg, d, k0, Q, Sg)
            else:
                raise ValueError

        if self.connecting_algo == 'TMM':
            result, T1 = transfer_1d_4(self.pol, ff_x, F, G, T, kz_top, kz_bot, self.theta, self.n_top, self.n_bot,
                                       type_complex=self.type_complex, use_pinv=self.use_pinv)
            self.T1 = T1  # Hurdle for jitting. This is not saved.

        elif self.connecting_algo == 'SMM':
            raise ValueError
            # de_ri, de_ti = scattering_1d_3(Wt, Wg, Vt, Vg, Sg, ff, Wr, self.fto, Kzr, Kzt,
            #                                self.n_top, self.n_bot, self.theta, self.pol)
        else:
            raise ValueError

        # return de_ri, de_ti, self.layer_info_list, self.T1
        return result

    @jax_device_set
    def solve_1d_conical(self, wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all):

        self.layer_info_list = []
        self.T1 = None

        ff_x = self.fto[0] * 2 + 1
        ff_y = 1

        k0 = 2 * jnp.pi / wavelength
        kx, ky = self.get_kx_ky_vector(wavelength)

        if self.connecting_algo == 'TMM':
            # Kx, ky, k_I_z, k_II_z, varphi, Y_I, Y_II, Z_I, Z_II, big_F, big_G, big_T \
            #     = transfer_1d_conical_1(ff, k0, self.n_top, self.n_bot, self.kx, self.theta, self.phi,
            #                             type_complex=self.type_complex)
            kz_top, kz_bot, varphi, big_F, big_G, big_T \
                = transfer_1d_conical_1(kx, ky, self.n_top, self.n_bot, type_complex=self.type_complex)

        elif self.connecting_algo == 'SMM':
            print('SMM for 1D conical is not implemented')
            return jnp.nan, jnp.nan
        else:
            raise ValueError

        for layer_index in range(len(self.thickness))[::-1]:

            epx_conv = epx_conv_all[layer_index]
            epy_conv = epy_conv_all[layer_index]
            epz_conv_i = epz_conv_i_all[layer_index]

            d = self.thickness[layer_index]

            if self.connecting_algo == 'TMM':
                # big_X, big_F, big_G, big_T, big_A_i, big_B, W_1, W_2, V_11, V_12, V_21, V_22, q_1, q_2 \
                #     = transfer_1d_conical_2(k0, Kx, ky, E_conv, E_conv_i, o_E_conv_i, ff, d,
                #                             varphi, big_F, big_G, big_T,
                #                             type_complex=self.type_complex, device=self.device)
                W, V, q = transfer_1d_conical_2(kx, ky, epx_conv, epy_conv, epz_conv_i, type_complex=self.type_complex,
                                                perturbation=self.perturbation, device=self.device,
                                                use_pinv=self.use_pinv)

                big_X, big_F, big_G, big_T, big_A_i, big_B, \
                    = transfer_1d_conical_3(k0, W, V, q, d, varphi, big_F, big_G, big_T, type_complex=self.type_complex,
                                            use_pinv=self.use_pinv)

                layer_info = [epz_conv_i, W, V, q, d, big_A_i, big_B]
                self.layer_info_list.append(layer_info)

            elif self.connecting_algo == 'SMM':
                raise ValueError
            else:
                raise ValueError

        if self.connecting_algo == 'TMM':
            # de_ri, de_ti, big_T1 = transfer_1d_conical_3(big_F, big_G, big_T, Z_I, Y_I, self.psi, self.theta, ff,
            #                                              delta_i0, k_I_z, k0, self.n_top, self.n_bot, k_II_z,
            #                                              type_complex=self.type_complex)
            result, big_T1 = transfer_1d_conical_4(ff_x, ff_y, big_F, big_G, big_T, kz_top, kz_bot, self.psi,
                                                   self.theta, self.n_top, self.n_bot, type_complex=self.type_complex,
                                                   use_pinv=self.use_pinv)
            self.T1 = big_T1

        elif self.connecting_algo == 'SMM':
            raise ValueError
        else:
            raise ValueError

        return result

    @jax_device_set
    # @jax.jit
    def solve_2d(self, wavelength, epx_conv_all, epy_conv_all, epz_conv_i_all):

        self.layer_info_list = []
        self.T1 = None

        ff_x = self.fto[0] * 2 + 1
        ff_y = self.fto[1] * 2 + 1

        k0 = 2 * jnp.pi / wavelength
        kx, ky = self.get_kx_ky_vector(wavelength)

        if self.connecting_algo == 'TMM':
            kz_top, kz_bot, varphi, big_F, big_G, big_T \
                = transfer_2d_1(kx, ky, self.n_top, self.n_bot, type_complex=self.type_complex)

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
                W, V, q = transfer_2d_2(kx, ky, epx_conv, epy_conv, epz_conv_i, self.type_complex, self.perturbation,
                                        use_pinv=self.use_pinv)

                big_X, big_F, big_G, big_T, big_A_i, big_B, \
                    = transfer_2d_3(k0, W, V, q, d, varphi, big_F, big_G, big_T, type_complex=self.type_complex,
                                    use_pinv=self.use_pinv)

                layer_info = [epz_conv_i, W, V, q, d, big_A_i, big_B]
                self.layer_info_list.append(layer_info)

            elif self.connecting_algo == 'SMM':
                raise ValueError
                # W, V, q = scattering_2d_wv(ff_xy, Kx, Ky, E_conv, o_E_conv, o_E_conv_i, E_conv_i)
                # A, B, Sl_dict, Sg_matrix, Sg = scattering_2d_2(W, Wg, V, Vg, d, k0, Sg, q)
            else:
                raise ValueError

        if self.connecting_algo == 'TMM':
            result, big_T1 = transfer_2d_4(ff_x, ff_y, big_F, big_G, big_T, kz_top, kz_bot, self.psi, self.theta,
                                           self.n_top, self.n_bot, type_complex=self.type_complex,
                                           use_pinv=self.use_pinv)
            self.T1 = big_T1

        elif self.connecting_algo == 'SMM':
            raise ValueError
            # de_ri, de_ti = scattering_2d_3(ff_xy, Wt, Wg, Vt, Vg, Sg, Wr, Kx, Ky, Kzr, Kzt, kz_inc, self.n_top,
            #                                self.pol, self.theta, self.phi, self.fto)
        else:
            raise ValueError
        # de_ri = de_ri.reshape((ff_y, ff_x)).T
        # de_ti = de_ti.reshape((ff_y, ff_x)).T

        # return de_ri, de_ti, self.layer_info_list, self.T1
        return result
