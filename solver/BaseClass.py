import copy
import time
import scipy
import scipy.io

import matplotlib.pyplot as plt
from pathlib import Path

from solver.convolution_matrix import *

from RCWA_functions import K_matrix as km
from RCWA_functions import PQ_matrices as pq
from TMM_functions import eigen_modes as em
from TMM_functions import scatter_matrices as sm
from RCWA_functions import redheffer_star as rs
from RCWA_functions import rcwa_initial_conditions as ic
from RCWA_functions import homogeneous_layer as hl
from scattering_method import *
from transfer_method import *


class RcwaBackbone:
    def __init__(self, grating_type, n_I=1, n_II=1, theta=0, phi=0, psi=0, fourier_order=10,
                 period=0.7, wls=np.linspace(0.5, 2.3, 400), pol=0,
                 patterns=None, thickness=None, algo='TMM'):

        self.grating_type = grating_type  # 1D=0, 1D_conical=1, 2D=2
        self.n_I = n_I
        self.n_II = n_II

        self.theta = theta * np.pi / 180
        self.phi = phi * np.pi / 180
        self.psi = psi * np.pi / 180

        self.fourier_order = fourier_order
        self.ff = 2 * self.fourier_order + 1
        self.period = period

        self.wls = wls

        self.pol = pol  # TE 0, TM 1

        # permittivity in grating layer
        # self.patterns = [[3.48, 1, 0.3]] if patterns is None else patterns
        self.patterns = [['SILICON', 1, 0.3]] if patterns is None else patterns
        self.thickness = [0.46] if thickness is None else thickness

        self.algo = algo

        # spectrum dimension
        # TODO: need to keep these result?
        if grating_type in (0, 1):
            self.spectrum_r = np.ndarray((len(wls), 2 * fourier_order + 1))
            self.spectrum_t = np.ndarray((len(wls), 2 * fourier_order + 1))
        elif grating_type == 2:
            self.spectrum_r = np.ndarray((len(wls), 2 * fourier_order + 1, 2 * fourier_order + 1))
            self.spectrum_t = np.ndarray((len(wls), 2 * fourier_order + 1, 2 * fourier_order + 1))
        else:
            raise ValueError

    def run(self):
        if self.grating_type == 0:
            self.lalanne_1d()
        elif self.grating_type == 1:
            self.lalanne_1d_conical()
        elif self.grating_type == 2:
            self.lalanne_2d()
        else:
            raise ValueError

    def plot(self):
        if self.grating_type == 0:
            plt.plot(res.wls, res.spectrum_r.sum(axis=1))
            plt.plot(res.wls, res.spectrum_t.sum(axis=1))
            plt.show()
        elif self.grating_type == 1:
            plt.plot(res.wls, res.spectrum_r.sum(axis=1))
            plt.plot(res.wls, res.spectrum_t.sum(axis=1))
            plt.show()
        elif self.grating_type == 2:
            plt.plot(res.wls, res.spectrum_r.sum(axis=(1, 2)))
            plt.plot(res.wls, res.spectrum_t.sum(axis=(1, 2)))
            plt.show()
        else:
            raise ValueError

    def lalanne_1d(self):

        fourier_indices = np.arange(-self.fourier_order, self.fourier_order + 1)

        delta_i0 = np.zeros(self.ff)
        delta_i0[self.fourier_order] = 1

        for i, wl in enumerate(self.wls):
            k0 = 2 * np.pi / wl

            E_conv_all = permittivity_mapping(self.patterns, wl, self.period, self.fourier_order)
            if self.pol == 0:  # TE
                oneover_E_conv_all = np.zeros(len(E_conv_all))  # Dummy for TE case
            elif self.pol == 1:  # TM
                oneover_E_conv_all = permittivity_mapping(self.patterns, wl, self.period, self.fourier_order,
                                                          oneover=True)
            else:
                raise ValueError

            # --------------------------------------------------------------------
            if self.algo == 'TMM':
                Kx, k_I_z, k_II_z, Kx, f, YZ_I, g, inc_term, T \
                    = transfer_1d_1(self.ff, self.pol, k0, self.n_I, self.n_II,
                                    self.theta, delta_i0, self.fourier_order, fourier_indices, wl, period)
            elif self.algo == 'SMM':
                Kx, Wg, Vg, Kzg, Wr, Vr, Kzr, Wt, Vt, Kzt, Ar, Br, Sg = scattering_1d_1(k0, self.n_I, self.n_II,
                                                                                        self.theta, self.phi, fourier_indices, self.period, self.pol)
            else:
                raise ValueError
            # --------------------------------------------------------------------

            for E_conv, oneover_E_conv, d in zip(E_conv_all[::-1], oneover_E_conv_all[::-1], self.thickness[::-1]):
                if self.pol == 0:
                    A = Kx ** 2 - E_conv
                    eigenvalues, W = np.linalg.eig(A)
                    q = eigenvalues ** 0.5

                    Q = np.diag(q)
                    V = W @ Q

                elif self.pol == 1:
                    E_i = np.linalg.inv(E_conv)
                    B = Kx @ E_i @ Kx - np.eye(E_conv.shape[0])
                    oneover_E_conv_i = np.linalg.inv(oneover_E_conv)

                    eigenvalues, W = np.linalg.eig(oneover_E_conv_i @ B)
                    q = eigenvalues ** 0.5

                    Q = np.diag(q)
                    V = oneover_E_conv @ W @ Q

                else:
                    raise ValueError
                # --------------------------------------------------------------------
                if self.algo == 'TMM':
                    f, g, T = transfer_1d_2(k0, q, d, W, V, f, g, self.fourier_order, T)
                elif self.algo == 'SMM':
                    A, B, S_dict, Sg = scattering_1d_2(W, Wg, V, Vg, d, k0, Q, Sg)

            if self.algo == 'TMM':
                de_ri, de_ti = transfer_1d_3(g, YZ_I, f, delta_i0, inc_term, T, k_I_z, k0, self.n_I, self.n_II, self.theta,
                                             self.pol, k_II_z)
            elif self.algo == 'SMM':
                de_ri, de_ti = scattering_1d_3(Wt, Wg, Vt, Vg, Sg, self.ff, Wr, self.fourier_order, Kzr, Kzt,
                                               self.n_I, self.n_II, self.theta, self.pol, Kx)
            else:
                raise ValueError

            self.spectrum_r[i] = de_ri
            self.spectrum_t[i] = de_ti

        return self.spectrum_r, self.spectrum_t

    def lalanne_1d_conical(self):

        # if self.theta == 0:
        #     self.theta = 0.001

        # pmtvy = draw_1d(self.patterns)
        # E_conv_all = to_conv_mat(pmtvy, self.fourier_order)

        fourier_indices = np.arange(-self.fourier_order, self.fourier_order + 1)

        delta_i0 = np.zeros(self.ff).reshape((-1, 1))
        delta_i0[self.fourier_order] = 1

        I = np.eye(self.ff)
        O = np.zeros((self.ff, self.ff))

        for i, wl in enumerate(self.wls):
            k0 = 2 * np.pi / wl

            E_conv_all = permittivity_mapping(self.patterns, wl, self.period, self.fourier_order)
            oneover_E_conv_all = permittivity_mapping(self.patterns, wl, self.period, self.fourier_order, oneover=True)

            kx_vector = k0 * (
                        self.n_I * np.sin(self.theta) * np.cos(self.phi) - fourier_indices * (wl / self.period)).astype(
                'complex')
            ky = k0 * self.n_I * np.sin(self.theta) * np.sin(self.phi)

            k_I_z = (k0 ** 2 * self.n_I ** 2 - kx_vector ** 2 - ky ** 2) ** 0.5
            k_II_z = (k0 ** 2 * self.n_II ** 2 - kx_vector ** 2 - ky ** 2) ** 0.5

            k_I_z = k_I_z.conjugate()
            k_II_z = k_II_z.conjugate()

            varphi = np.arctan(ky / kx_vector)

            Y_I = np.diag(k_I_z / k0)
            Y_II = np.diag(k_II_z / k0)

            Z_I = np.diag(k_I_z / (k0 * self.n_I ** 2))
            Z_II = np.diag(k_II_z / (k0 * self.n_II ** 2))

            Kx = np.diag(kx_vector / k0)

            big_F = np.block([[I, O], [O, 1j * Z_II]])
            big_G = np.block([[1j * Y_II, O], [O, I]])

            big_T = np.eye(2 * self.ff)

            # for E_conv, d in zip(E_conv_all[::-1], self.thickness[::-1]):
            for E_conv, oneover_E_conv, d in zip(E_conv_all[::-1], oneover_E_conv_all[::-1], self.thickness[::-1]):
                E_i = np.linalg.inv(E_conv)
                oneover_E_conv_i = np.linalg.inv(oneover_E_conv)

                A = Kx ** 2 - E_conv
                B = Kx @ E_i @ Kx - I
                A_i = np.linalg.inv(A)
                B_i = np.linalg.inv(B)

                to_decompose_W_1 = ky ** 2 * I + A
                to_decompose_W_2 = ky ** 2 * I + B @ oneover_E_conv_i

                # TODO: using eigh
                eigenvalues_1, W_1 = np.linalg.eig(to_decompose_W_1)
                eigenvalues_2, W_2 = np.linalg.eig(to_decompose_W_2)

                q_1 = eigenvalues_1 ** 0.5
                q_2 = eigenvalues_2 ** 0.5

                Q_1 = np.diag(q_1)
                Q_2 = np.diag(q_2)

                V_11 = A_i @ W_1 @ Q_1
                V_12 = (ky / k0) * A_i @ Kx @ W_2
                V_21 = (ky / k0) * B_i @ Kx @ E_i @ W_1
                V_22 = B_i @ W_2 @ Q_2

                X_1 = np.diag(np.exp(-k0 * q_1 * d))
                X_2 = np.diag(np.exp(-k0 * q_2 * d))

                F_c = np.diag(np.cos(varphi))
                F_s = np.diag(np.sin(varphi))

                V_ss = F_c @ V_11
                V_sp = F_c @ V_12 - F_s @ W_2
                W_ss = F_c @ W_1 + F_s @ V_21
                W_sp = F_s @ V_22
                W_ps = F_s @ V_11
                W_pp = F_c @ W_2 + F_s @ V_12
                V_ps = F_c @ V_21 - F_s @ W_1
                V_pp = F_c @ V_22

                big_I = np.eye(2 * (len(I)))
                big_X = np.block([[X_1, O], [O, X_2]])
                big_W = np.block([[V_ss, V_sp], [W_ps, W_pp]])
                big_V = np.block([[W_ss, W_sp], [V_ps, V_pp]])

                big_W_i = np.linalg.inv(big_W)
                big_V_i = np.linalg.inv(big_V)

                big_A = 0.5 * (big_W_i @ big_F + big_V_i @ big_G)
                big_B = 0.5 * (big_W_i @ big_F - big_V_i @ big_G)

                big_A_i = np.linalg.inv(big_A)

                big_F = big_W @ (big_I + big_X @ big_B @ big_A_i @ big_X)
                big_G = big_V @ (big_I - big_X @ big_B @ big_A_i @ big_X)

                big_T = big_T @ big_A_i @ big_X

            big_F_11 = big_F[:self.ff, :self.ff]
            big_F_12 = big_F[:self.ff, self.ff:]
            big_F_21 = big_F[self.ff:, :self.ff]
            big_F_22 = big_F[self.ff:, self.ff:]

            big_G_11 = big_G[:self.ff, :self.ff]
            big_G_12 = big_G[:self.ff, self.ff:]
            big_G_21 = big_G[self.ff:, :self.ff]
            big_G_22 = big_G[self.ff:, self.ff:]

            # Final Equation in form of AX=B
            final_A = np.block(
                [
                    [I, O, -big_F_11, -big_F_12],
                    [O, -1j * Z_I, -big_F_21, -big_F_22],
                    [-1j * Y_I, O, -big_G_11, -big_G_12],
                    [O, I, -big_G_21, -big_G_22],
                ]
            )

            final_B = np.block([
                [-np.sin(self.psi) * delta_i0],
                [-np.cos(self.psi) * np.cos(self.theta) * delta_i0],
                [-1j * np.sin(self.psi) * self.n_I * np.cos(self.theta) * delta_i0],
                [1j * self.n_I * np.cos(self.psi) * delta_i0]
            ]
            )

            final_X = np.linalg.inv(final_A) @ final_B

            R_s = final_X[:self.ff, :].flatten()
            R_p = final_X[self.ff:2 * self.ff, :].flatten()

            big_T = big_T @ final_X[2 * self.ff:, :]
            T_s = big_T[:self.ff, :].flatten()
            T_p = big_T[self.ff:, :].flatten()

            DEri = R_s * np.conj(R_s) * np.real(k_I_z / (k0 * self.n_I * np.cos(self.theta))) \
                   + R_p * np.conj(R_p) * np.real((k_I_z / self.n_I ** 2) / (k0 * self.n_I * np.cos(self.theta)))

            DEti = T_s * np.conj(T_s) * np.real(k_II_z / (k0 * self.n_I * np.cos(self.theta))) \
                   + T_p * np.conj(T_p) * np.real((k_II_z / self.n_II ** 2) / (k0 * self.n_I * np.cos(self.theta)))

            # self.spectrum_r.append(DEri.sum())
            # self.spectrum_t.append(DEti.sum())

            self.spectrum_r[i] = DEri.real
            self.spectrum_t[i] = DEti.real

        return self.spectrum_r, self.spectrum_t

    def lalanne_2d(self):

        fourier_indices = np.arange(-self.fourier_order, self.fourier_order + 1)

        delta_i0 = np.zeros((self.ff ** 2, 1))
        delta_i0[self.ff ** 2 // 2, 0] = 1

        I = np.eye(self.ff ** 2)
        O = np.zeros((self.ff ** 2, self.ff ** 2))

        center = self.ff ** 2

        for i, wl in enumerate(self.wls):
            k0 = 2 * np.pi / wl

            E_conv_all = permittivity_mapping(self.patterns, wl, self.period, self.fourier_order)
            oneover_E_conv_all = permittivity_mapping(self.patterns, wl, self.period, self.fourier_order, oneover=True)

            if self.algo == 'TMM':
                Kx, Ky, k_I_z, k_II_z, varphi, Y_I, Y_II, Z_I, Z_II, big_F, big_G, big_T\
                    = transfer_2d_1(self.ff, k0, self.n_I, self.n_II, self.period, fourier_indices, self.theta, self.phi, wl)
            elif self.algo == 'SMM':
                Kx, Ky, kz_inc, Wg, Vg, Kzg, Wr, Vr, Kzr, Wt, Vt, Kzt, Ar, Br, Sg\
                    = scattering_2d_1(self.n_I, self.n_II, self.theta, self.phi, k0, self.period, self.fourier_order)
            else:
                raise ValueError

            # TODO: MERGE
            for E_conv, oneover_E_conv, d in zip(E_conv_all[::-1], oneover_E_conv_all[::-1], self.thickness[::-1]):
                E_i = np.linalg.inv(E_conv)
                oneover_E_conv_i = np.linalg.inv(oneover_E_conv)

                if self.algo == 'TMM':
                    W, V, LAMBDA, Lambda = transfer_2d_wv(self.ff, Kx, E_i, Ky, oneover_E_conv_i, E_conv, center)
                elif self.algo == 'SMM':
                    # aa=np.abs(U1_from_S - Q).sum()
                    # bb=np.abs(S2_from_S - Gamma_squared).sum()
                    # print(aa, bb)
                    # TODO: why aa == 0 but bb != 0?
                    # -------------------------
                    W, V, LAMBDA = scattering_2d_wv(self.ff, Kx, Ky, E_conv, oneover_E_conv, oneover_E_conv_i, E_i)
                else:
                    raise ValueError

                if self.algo == 'TMM':
                    big_F, big_G, big_T = transfer_2d_2(k0, d, W, V, center, Lambda, varphi, I, O, big_F, big_G, big_T)
                elif self.algo == 'SMM':
                    A, B, Sl_dict, Sg_matrix, Sg = scattering_2d_2(W, Wg, V, Vg, d, k0, Sg, LAMBDA)
                else:
                    raise ValueError

            if self.algo == 'TMM':
                de_ri, de_ti = transfer_2d_3(center, big_F, big_G, big_T, Z_I, Y_I, self.psi, self.theta, self.ff, delta_i0,
                                             k_I_z, k0, self.n_I, self.n_II, k_II_z)

            elif self.algo == 'SMM':

                de_ri, de_ti = scattering_2d_3(Wt, Wg, Vt, Vg, Sg, Wr, Kx, Ky, Kzr, Kzt, kz_inc, self.n_I,
                                               self.pol, self.theta, self.phi, self.fourier_order, self.ff)
            else:
                raise ValueError

            self.spectrum_r[i] = de_ri.reshape((self.ff, self.ff)).real
            self.spectrum_t[i] = de_ti.reshape((self.ff, self.ff)).real

        return self.spectrum_r, self.spectrum_t


if __name__ == '__main__':
    n_I = 4
    n_II = 20

    theta = 30
    phi = 60

    fourier_order = 2
    wls = np.linspace(500, 2300, 10)

    grating_type = 2

    if grating_type == 0:
        period = [700]
        phi = 0

    elif grating_type == 2:
        period = [700, 700]

    # TODO: integrate psi into this
    polarization = 1  # TE 0, TM 1

    if polarization == 0:
        psi = 90
    elif polarization == 1:
        psi = 0

    # permittivity in grating layer
    patterns = [[3.48, 1, 0.3], [3.48, 1, 0.3]]  # n_ridge, n_groove, fill_factor
    thickness = [460, 660]

    t0 = time.time()
    res = RcwaBackbone(grating_type, n_I, n_II, theta, phi, psi, fourier_order, period, wls,
                       polarization, patterns, thickness, algo='TMM')
    res.run()
    print(time.time() - t0)
    res.plot()

    t0 = time.time()
    res = RcwaBackbone(grating_type, n_I, n_II, theta, phi, psi, fourier_order, period, wls,
                       polarization, patterns, thickness, algo='SMM')

    res.run()
    print(time.time() - t0)
    res.plot()
