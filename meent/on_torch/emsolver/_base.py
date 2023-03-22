import torch

import numpy as np

from copy import deepcopy

from .primitives import Eig
from .scattering_method import scattering_1d_1, scattering_1d_2, scattering_1d_3, scattering_2d_1, scattering_2d_wv, \
    scattering_2d_2, scattering_2d_3
from .transfer_method import transfer_1d_1, transfer_1d_2, transfer_1d_3, transfer_1d_conical_1, transfer_1d_conical_2, \
    transfer_1d_conical_3, transfer_2d_1, transfer_2d_wv, transfer_2d_2, transfer_2d_3


class _BaseRCWA:
    def __init__(self, grating_type, n_I=1., n_II=1., theta=0., phi=0., psi=0., pol=0, fourier_order=(2, 2),
                 period=(100, 100), wavelength=900,
                 thickness=None, algo='TMM', perturbation=1E-10,
                 device='cpu', type_complex=torch.complex128):

        self.device = device
        self.type_complex = type_complex

        # common
        self.grating_type = grating_type  # 1D=0, 1D_conical=1, 2D=2

        self.n_I = n_I
        self.n_II = n_II

        # degree to radian due to JAX JIT
        self.theta = torch.tensor(theta)
        self.phi = torch.tensor(phi)
        self.psi = torch.tensor(psi)  # TODO: integrate psi and pol

        self.pol = pol  # TE 0, TM 1
        if self.pol == 0:  # TE
            self.psi = torch.tensor(np.pi / 2, device=self.device)
        elif self.pol == 1:  # TM
            self.psi = torch.tensor(0, device=self.device)
        else:
            print('not implemented yet')
            raise ValueError

        if type(fourier_order) == int:
            self._fourier_order = [fourier_order, 0]
        elif len(fourier_order) == 1:
            self._fourier_order = list(fourier_order) + [0]
        else:
            self._fourier_order = [int(v) for v in fourier_order]

        self.period = deepcopy(period)

        self.wavelength = wavelength
        self.thickness = deepcopy(thickness)

        self.algo = algo
        self.perturbation = perturbation

        self.layer_info_list = []
        self.T1 = None

        self.theta = torch.where(self.theta == 0, self.perturbation, self.theta)  # TODO: check correct?

        self.kx_vector = None

    @property
    def fourier_order(self):
        return self._fourier_order

    @fourier_order.setter
    def fourier_order(self, fourier_order):
        if type(fourier_order) == int:
            self._fourier_order = [fourier_order, 0]
        elif len(fourier_order) == 1:
            self._fourier_order = list(fourier_order) + [0]
        else:
            self._fourier_order = [int(v) for v in fourier_order]

    def get_kx_vector(self, wavelength):

        k0 = 2 * np.pi / wavelength
        fourier_indices = torch.arange(-self.fourier_order[0], self.fourier_order[0] + 1, device=self.device)
        if self.grating_type == 0:
            kx_vector = k0 * (self.n_I * torch.sin(self.theta) + fourier_indices * (wavelength / self.period[0])
                              ).type(self.type_complex)
        else:
            kx_vector = k0 * (self.n_I * torch.sin(self.theta) * torch.cos(self.phi) + fourier_indices * (
                    wavelength / self.period[0])).type(self.type_complex)

        # kx_vector = torch.where(kx_vector == 0, self.perturbation, kx_vector)

        return kx_vector

    def solve_1d(self, wavelength, E_conv_all, o_E_conv_all):

        self.layer_info_list = []
        self.T1 = None

        # fourier_indices = torch.arange(-self.fourier_order, self.fourier_order + 1, device=self.device)

        ff = self.fourier_order[0] * 2 + 1

        delta_i0 = torch.zeros(ff, device=self.device, dtype=self.type_complex)
        delta_i0[self.fourier_order[0]] = 1

        k0 = 2 * np.pi / wavelength

        if self.algo == 'TMM':
            kx_vector, Kx, k_I_z, k_II_z, f, YZ_I, g, inc_term, T \
                = transfer_1d_1(ff, self.pol, k0, self.n_I, self.n_II, self.kx_vector,
                                self.theta, delta_i0, self.fourier_order,
                                device=self.device, type_complex=self.type_complex)
        elif self.algo == 'SMM':
            Kx, Wg, Vg, Kzg, Wr, Vr, Kzr, Wt, Vt, Kzt, Ar, Br, Sg \
                = scattering_1d_1(k0, self.n_I, self.n_II, self.theta, self.phi, fourier_indices, self.period,
                                  self.pol, wl=wavelength)
        else:
            raise ValueError

        count = min(len(E_conv_all), len(o_E_conv_all), len(self.thickness))

        # From the last layer
        for layer_index in np.arange(count-1, -1, -1):

            E_conv = E_conv_all[layer_index]
            o_E_conv = o_E_conv_all[layer_index]
            d = self.thickness[layer_index]

        # Can't use this. seems like bug in Torch.
        # for E_conv, o_E_conv, d in zip(E_conv_all[::-1], o_E_conv_all[::-1], self.thickness[::-1]):

            if self.pol == 0:
                E_conv_i = None
                A = Kx ** 2 - E_conv
                Eig.perturbation = self.perturbation
                eigenvalues, W = Eig.apply(A)
                q = eigenvalues ** 0.5

                Q = torch.diag(q)
                V = W @ Q

            elif self.pol == 1:
                E_conv_i = torch.linalg.inv(E_conv)
                B = Kx @ E_conv_i @ Kx - torch.eye(E_conv.shape[0], device=self.device, dtype=self.type_complex)
                o_E_conv_i = torch.linalg.inv(o_E_conv)

                Eig.perturbation = self.perturbation
                eigenvalues, W = Eig.apply(o_E_conv_i @ B)
                q = eigenvalues ** 0.5

                Q = torch.diag(q)
                V = o_E_conv @ W @ Q

            else:
                raise ValueError

            if self.algo == 'TMM':
                X, f, g, T, a_i, b = transfer_1d_2(k0, q, d, W, V, f, g, self.fourier_order, T,
                                                   device=self.device, type_complex=self.type_complex)

                layer_info = [E_conv_i, q, W, X, a_i, b, d]
                self.layer_info_list.append(layer_info)

            elif self.algo == 'SMM':
                A, B, S_dict, Sg = scattering_1d_2(W, Wg, V, Vg, d, k0, Q, Sg)
            else:
                raise ValueError

        if self.algo == 'TMM':
            de_ri, de_ti, T1 = transfer_1d_3(g, YZ_I, f, delta_i0, inc_term, T, k_I_z, k0, self.n_I, self.n_II,
                                             self.theta, self.pol, k_II_z)
            self.T1 = T1

        elif self.algo == 'SMM':
            de_ri, de_ti = scattering_1d_3(Wt, Wg, Vt, Vg, Sg, self.ff, Wr, self.fourier_order, Kzr, Kzt,
                                           self.n_I, self.n_II, self.theta, self.pol)
        else:
            raise ValueError

        return de_ri, de_ti, self.layer_info_list, self.T1

    def solve_1d_conical(self, wavelength, E_conv_all, o_E_conv_all):

        self.layer_info_list = []
        self.T1 = None

        # fourier_indices = torch.arange(-self.fourier_order, self.fourier_order + 1, device=self.device)
        ff = self.fourier_order[0] * 2 + 1

        delta_i0 = torch.zeros(ff, device=self.device, dtype=self.type_complex)
        delta_i0[self.fourier_order[0]] = 1

        k0 = 2 * np.pi / wavelength

        if self.algo == 'TMM':
            Kx, ky, k_I_z, k_II_z, varphi, Y_I, Y_II, Z_I, Z_II, big_F, big_G, big_T \
                = transfer_1d_conical_1(ff, k0, self.n_I, self.n_II, self.kx_vector, self.theta, self.phi,
                                        device=self.device, type_complex=self.type_complex)
        elif self.algo == 'SMM':
            print('SMM for 1D conical is not implemented')
            return np.nan, np.nan
        else:
            raise ValueError

        count = min(len(E_conv_all), len(o_E_conv_all), len(self.thickness))

        # From the last layer
        for layer_index in range(count)[::-1]:

            E_conv = E_conv_all[layer_index]
            o_E_conv = o_E_conv_all[layer_index]
            d = self.thickness[layer_index]

        # Can't use this. Seems like bug in Torch.
        # for E_conv, o_E_conv, d in zip(E_conv_all[::-1], o_E_conv_all[::-1], self.thickness[::-1]):
            E_conv_i = torch.linalg.inv(E_conv)
            o_E_conv_i = torch.linalg.inv(o_E_conv)

            if self.algo == 'TMM':
                big_X, big_F, big_G, big_T, big_A_i, big_B, W_1, W_2, V_11, V_12, V_21, V_22, q_1, q_2\
                    = transfer_1d_conical_2(k0, Kx, ky, E_conv, E_conv_i, o_E_conv_i, ff, d,
                                                            varphi, big_F, big_G, big_T,
                                                            device=self.device, type_complex=self.type_complex)

                layer_info = [E_conv_i, q_1, q_2, W_1, W_2, V_11, V_12, V_21, V_22, big_X, big_A_i, big_B, d]
                self.layer_info_list.append(layer_info)

            elif self.algo == 'SMM':
                raise ValueError
            else:
                raise ValueError

        if self.algo == 'TMM':
            de_ri, de_ti, big_T1 = transfer_1d_conical_3(big_F, big_G, big_T, Z_I, Y_I, self.psi, self.theta, ff,
                                                 delta_i0, k_I_z, k0, self.n_I, self.n_II, k_II_z,
                                                 device=self.device, type_complex=self.type_complex)
            self.T1 = big_T1

        elif self.algo == 'SMM':
            raise ValueError
        else:
            raise ValueError

        return de_ri, de_ti, self.layer_info_list, self.T1

    def solve_2d(self, wavelength, E_conv_all, o_E_conv_all):

        self.layer_info_list = []
        self.T1 = None

        # fourier_indices = torch.arange(-self.fourier_order, self.fourier_order + 1, device=self.device)
        fourier_indices_y = torch.arange(-self.fourier_order[1], self.fourier_order[1] + 1, device=self.device)

        ff_x = self.fourier_order[0] * 2 + 1
        ff_y = self.fourier_order[1] * 2 + 1
        ff_xy = ff_x * ff_y

        delta_i0 = torch.zeros((ff_xy, 1), device=self.device, dtype=self.type_complex)
        delta_i0[ff_xy // 2, 0] = 1

        I = torch.eye(ff_xy, device=self.device, dtype=self.type_complex)
        O = torch.zeros((ff_xy, ff_xy), device=self.device, dtype=self.type_complex)

        center = ff_xy

        k0 = 2 * np.pi / wavelength

        if self.algo == 'TMM':
            kx_vector, ky_vector, Kx, Ky, k_I_z, k_II_z, varphi, Y_I, Y_II, Z_I, Z_II, big_F, big_G, big_T \
                = transfer_2d_1(ff_x, ff_y, ff_xy, k0, self.n_I, self.n_II, self.kx_vector, self.period, fourier_indices_y,
                                self.theta, self.phi, wavelength, device=self.device, type_complex=self.type_complex)
        elif self.algo == 'SMM':
            Kx, Ky, kz_inc, Wg, Vg, Kzg, Wr, Vr, Kzr, Wt, Vt, Kzt, Ar, Br, Sg \
                = scattering_2d_1(self.n_I, self.n_II, self.theta, self.phi, k0, self.period, self.fourier_order)
        else:
            raise ValueError

        # for E_conv, o_E_conv, d in zip(E_conv_all[::-1], o_E_conv_all[::-1], self.thickness[::-1]):

        count = min(len(E_conv_all), len(o_E_conv_all), len(self.thickness))

        # From the last layer
        for layer_index in range(count)[::-1]:

            E_conv = E_conv_all[layer_index]
            o_E_conv = o_E_conv_all[layer_index]
            d = self.thickness[layer_index]
            E_conv_i = torch.linalg.inv(E_conv)
            o_E_conv_i = torch.linalg.inv(o_E_conv)

            if self.algo == 'TMM':
                W, V, q = transfer_2d_wv(ff_xy, Kx, E_conv_i, Ky, o_E_conv_i, E_conv,
                                         device=self.device, type_complex=self.type_complex)

                big_X, big_F, big_G, big_T, big_A_i, big_B, \
                W_11, W_12, W_21, W_22, V_11, V_12, V_21, V_22 \
                    = transfer_2d_2(k0, d, W, V, center, q, varphi, I, O, big_F, big_G, big_T, device=self.device,
                                    type_complex=self.type_complex)

                layer_info = [E_conv_i, q, W_11, W_12, W_21, W_22, V_11, V_12, V_21, V_22, big_X, big_A_i, big_B, d]
                self.layer_info_list.append(layer_info)

            elif self.algo == 'SMM':
                W, V, LAMBDA = scattering_2d_wv(Kx, Ky, E_conv, o_E_conv, o_E_conv_i, E_conv_i)
                A, B, Sl_dict, Sg_matrix, Sg = scattering_2d_2(W, Wg, V, Vg, d, k0, Sg, LAMBDA)
            else:
                raise ValueError

        if self.algo == 'TMM':
            de_ri, de_ti, big_T1 = transfer_2d_3(center, big_F, big_G, big_T, Z_I, Y_I, self.psi, self.theta, ff_xy,
                                                 delta_i0, k_I_z, k0, self.n_I, self.n_II, k_II_z, device=self.device,
                                                 type_complex=self.type_complex)
            self.T1 = big_T1

        elif self.algo == 'SMM':
            de_ri, de_ti = scattering_2d_3(Wt, Wg, Vt, Vg, Sg, Wr, Kx, Ky, Kzr, Kzt, kz_inc, self.n_I,
                                           self.pol, self.theta, self.phi, self.fourier_order)
        else:
            raise ValueError

        de_ri = de_ri.reshape((ff_y, ff_x))
        de_ti = de_ti.reshape((ff_y, ff_x))

        return de_ri, de_ti, self.layer_info_list, self.T1
