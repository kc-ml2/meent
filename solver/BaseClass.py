import copy

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

class RcwaBackbone:
    def __init__(self, grating_type, n_I=1, n_II=1, theta=0, phi=0, psi=0, fourier_order=10,
                 period=0.7, wls=np.linspace(0.5, 2.3, 400), polarization=0,
                 patterns=None, thickness=None, algo='TMM'):

        self.grating_type = grating_type  # 1D=0, 1D_conical=1, 2D=2 # TODO
        self.n_I = n_I
        self.n_II = n_II

        self.theta = theta * np.pi / 180
        self.phi = phi * np.pi / 180
        self.psi = psi * np.pi / 180

        self.fourier_order = fourier_order
        self.ff = 2 * self.fourier_order + 1
        self.period = period

        self.wls = wls

        self.polarization = polarization  # TE 0, TM 1

        # permittivity in grating layer
        # self.patterns = [[3.48, 1, 0.3]] if patterns is None else patterns
        self.patterns = [['SILICON', 1, 0.3]] if patterns is None else patterns
        self.thickness = [0.46] if thickness is None else thickness

        self.algo = algo

        # spectrum dimension
        # TODO: need to keep these result?
        if grating_type in (0, 1):
            self.spectrum_r = np.ndarray((len(wls), 2*fourier_order+1))
            self.spectrum_t = np.ndarray((len(wls), 2*fourier_order+1))
        elif grating_type == 2:
            self.spectrum_r = np.ndarray((len(wls), 2*fourier_order+1, 2*fourier_order+1))
            self.spectrum_t = np.ndarray((len(wls), 2*fourier_order+1, 2*fourier_order+1))
        else:
            raise ValueError

    def lalanne_1d(self):

        fourier_indices = np.arange(-self.fourier_order, self.fourier_order + 1)

        delta_i0 = np.zeros(2 * self.fourier_order + 1)
        delta_i0[self.fourier_order] = 1

        for i, wl in enumerate(self.wls):
            k0 = 2 * np.pi / wl

            E_conv_all = permittivity_mapping(self.patterns, wl, self.period, self.fourier_order)
            if self.polarization == 0:  # TE
                oneover_E_conv_all = np.zeros(len(E_conv_all))  # Dummy for TE case

            elif self.polarization == 1:  # TM
                oneover_E_conv_all = permittivity_mapping(self.patterns, wl, self.period, self.fourier_order, oneover=True)

            else:
                raise ValueError

            kx_vector = k0 * (self.n_I * np.sin(self.theta) - fourier_indices * (wl / self.period)).astype('complex')

            k_I_z = (k0 ** 2 * self.n_I ** 2 - kx_vector ** 2) ** 0.5
            k_II_z = (k0 ** 2 * self.n_II ** 2 - kx_vector ** 2) ** 0.5

            k_I_z = k_I_z.conjugate()
            k_II_z = k_II_z.conjugate()

            Kx = np.diag(kx_vector / k0)
            # --------------------------------------------------------------------
            if self.algo == 'TMM':
                f, YZ_I, g, inc_term, T = TMM1(self.ff, self.polarization, k_I_z, k0, k_II_z, self.n_I,
                                               self.theta, delta_i0, self.fourier_order)
            elif self.algo == 'SMM':
                Wg, Vg, Kzg, Wr, Vr, Kzr, Wt, Vt, Kzt, Ar, Br, Sg = SMM1(Kx, k0, self.n_I, self.n_II)
            # --------------------------------------------------------------------

            for E_conv, oneover_E_conv, d in zip(E_conv_all[::-1], oneover_E_conv_all[::-1], self.thickness[::-1]):
                if self.polarization == 0:
                    A = Kx ** 2 - E_conv
                    eigenvalues, W = np.linalg.eig(A)
                    q = eigenvalues ** 0.5

                    Q = np.diag(q)
                    V = W @ Q

                elif self.polarization == 1:
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
                    f, g, T = TMM2(k0, q, d, W, V, f, g, self.fourier_order, T)
                elif self.algo == 'SMM':
                    A, B, S_dict, Sg = SMM2(W, Wg, V, Vg, d, k0, Q, Sg)

            if self.algo == 'TMM':
                de_ri, de_ti = TMM3(g, YZ_I, f, delta_i0, inc_term, T, k_I_z, k0, self.n_I, self.theta, self.polarization, k_II_z)
            elif self.algo == 'SMM':
                de_ri, de_ti = SMM3(Wt, Wg, Vt, Vg, Sg, self.ff, Wr, self.fourier_order, Kzr, k0, k_I_z, Kzt, k_II_z)

            self.spectrum_r[i] = de_ri
            self.spectrum_t[i] = de_ti

        return self.spectrum_r, self.spectrum_t

    def lalanne_1d_conical(self):

        if self.theta == 0:
            self.theta = 0.001

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

            kx_vector = k0 * (self.n_I * np.sin(self.theta) * np.cos(self.phi) - fourier_indices * (wl / self.period)).astype('complex')
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

        if self.theta == 0:
            self.theta = 1E-10

        # pmtvy = draw_1d(self.patterns)
        # E_conv_all = to_conv_mat(pmtvy, self.fourier_order)

        fourier_indices = np.arange(-self.fourier_order, self.fourier_order + 1)

        delta_i0 = np.zeros(self.ff**2).reshape((-1, 1))
        delta_i0[self.ff ** 2 // 2, 0] = 1

        I = np.eye(self.ff ** 2)
        O = np.zeros((self.ff ** 2, self.ff ** 2))

        center = self.ff ** 2

        for i, wl in enumerate(self.wls):

            E_conv_all = permittivity_mapping(self.patterns, wl, self.period, self.fourier_order)
            oneover_E_conv_all = permittivity_mapping(self.patterns, wl, self.period, self.fourier_order, oneover=True)

            k0 = 2 * np.pi / wl

            kx_vector = k0 * (self.n_I * np.sin(self.theta) * np.cos(self.phi) - fourier_indices * (wl / self.period[0])).astype('complex')
            ky_vector = k0 * (self.n_I * np.sin(self.theta) * np.sin(self.phi) - fourier_indices * (wl / self.period[1])).astype('complex')

            Kx = np.diag(np.tile(kx_vector, self.ff).flatten()) / k0
            Ky = np.diag(np.tile(ky_vector.reshape((-1, 1)), self.ff).flatten()) / k0

            k_I_z = (k0 ** 2 * self.n_I ** 2 - kx_vector ** 2 - ky_vector.reshape((-1, 1)) ** 2) ** 0.5
            k_II_z = (k0 ** 2 * self.n_II ** 2 - kx_vector ** 2 - ky_vector.reshape((-1, 1)) ** 2) ** 0.5

            k_I_z = k_I_z.flatten().conjugate()
            k_II_z = k_II_z.flatten().conjugate()

            varphi = np.arctan(ky_vector.reshape((-1, 1)) / kx_vector).flatten()

            Y_I = np.diag(k_I_z / k0)
            Y_II = np.diag(k_II_z / k0)

            Z_I = np.diag(k_I_z / (k0 * self.n_I ** 2))
            Z_II = np.diag(k_II_z / (k0 * self.n_II ** 2))

            big_F = np.block([[I, O], [O, 1j * Z_II]])
            big_G = np.block([[1j * Y_II, O], [O, I]])

            big_T = np.eye(self.ff ** 2 * 2)

            for E_conv, oneover_E_conv, d in zip(E_conv_all[::-1], oneover_E_conv_all[::-1], self.thickness[::-1]):
                E_i = np.linalg.inv(E_conv)

                B = Kx @ E_i @ Kx - I
                D = Ky @ E_i @ Ky - I
                oneover_E_conv_i = np.linalg.inv(oneover_E_conv)

                S2_from_S = np.block(
                    [
                        [Ky ** 2 + B @ oneover_E_conv_i, Kx @ (E_i @ Ky @ E_conv - Ky)],
                        [Ky @ (E_i @ Kx @ oneover_E_conv_i - Kx), Kx ** 2 + D @ E_conv]
                    ])

                # TODO: using eigh
                eigenvalues, W = np.linalg.eig(S2_from_S)

                q = eigenvalues ** 0.5

                q_1 = q[:center]
                q_2 = q[center:]

                Q = np.diag(q)
                Q_i = np.linalg.inv(Q)
                U1_from_S = np.block(
                    [
                        [-Kx @ Ky, Kx ** 2 - E_conv],
                        [oneover_E_conv_i - Ky ** 2, Ky @ Kx]  # TODO Check x y order
                    ]
                )
                V = U1_from_S @ W @ Q_i

                W_11 = W[:center, :center]
                W_12 = W[:center, center:]
                W_21 = W[center:, :center]
                W_22 = W[center:, center:]

                V_11 = V[:center, :center]
                V_12 = V[:center, center:]
                V_21 = V[center:, :center]
                V_22 = V[center:, center:]

                X_1 = np.diag(np.exp(-k0 * q_1 * d))
                X_2 = np.diag(np.exp(-k0 * q_2 * d))

                F_c = np.diag(np.cos(varphi))
                F_s = np.diag(np.sin(varphi))

                W_ss = F_c @ W_21 - F_s @ W_11
                W_sp = F_c @ W_22 - F_s @ W_12
                W_ps = F_c @ W_11 + F_s @ W_21
                W_pp = F_c @ W_12 + F_s @ W_22

                V_ss = F_c @ V_11 + F_s @ V_21
                V_sp = F_c @ V_12 + F_s @ V_22
                V_ps = F_c @ V_21 - F_s @ V_11
                V_pp = F_c @ V_22 - F_s @ V_12

                big_I = np.eye(2 * (len(I)))
                big_X = np.block([[X_1, O], [O, X_2]])
                big_W = np.block([[W_ss, W_sp], [W_ps, W_pp]])
                big_V = np.block([[V_ss, V_sp], [V_ps, V_pp]])

                big_W_i = np.linalg.inv(big_W)
                big_V_i = np.linalg.inv(big_V)

                big_A = 0.5 * (big_W_i @ big_F + big_V_i @ big_G)
                big_B = 0.5 * (big_W_i @ big_F - big_V_i @ big_G)

                big_A_i = np.linalg.inv(big_A)

                big_F = big_W @ (big_I + big_X @ big_B @ big_A_i @ big_X)
                big_G = big_V @ (big_I - big_X @ big_B @ big_A_i @ big_X)

                big_T = big_T @ big_A_i @ big_X


            big_F_11 = big_F[:center, :center]
            big_F_12 = big_F[:center, center:]
            big_F_21 = big_F[center:, :center]
            big_F_22 = big_F[center:, center:]

            big_G_11 = big_G[:center, :center]
            big_G_12 = big_G[:center, center:]
            big_G_21 = big_G[center:, :center]
            big_G_22 = big_G[center:, center:]

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

            R_s = final_X[:self.ff ** 2, :].flatten()
            R_p = final_X[self.ff ** 2:2 * self.ff ** 2, :].flatten()

            big_T = big_T @ final_X[2 * self.ff ** 2:, :]
            T_s = big_T[:self.ff ** 2, :].flatten()
            T_p = big_T[self.ff ** 2:, :].flatten()

            DEri = R_s * np.conj(R_s) * np.real(k_I_z / (k0 * self.n_I * np.cos(self.theta))) \
                   + R_p * np.conj(R_p) * np.real((k_I_z / self.n_I ** 2) / (k0 * self.n_I * np.cos(self.theta)))

            DEti = T_s * np.conj(T_s) * np.real(k_II_z / (k0 * self.n_I * np.cos(self.theta))) \
                   + T_p * np.conj(T_p) * np.real((k_II_z / self.n_II ** 2) / (k0 * self.n_I * np.cos(self.theta)))

            # self.spectrum_r.append(DEri.sum())
            # self.spectrum_t.append(DEti.sum())

            self.spectrum_r[i] = DEri.reshape((self.ff, -1)).real
            self.spectrum_t[i] = DEti.reshape((self.ff, -1)).real

        return self.spectrum_r, self.spectrum_t


def TMM1(ff, polarization, k_I_z, k0, k_II_z, n_I, theta, delta_i0, fourier_order):
    f = np.eye(ff)

    if polarization == 0:  # TE
        Y_I = np.diag(k_I_z / k0)
        Y_II = np.diag(k_II_z / k0)

        YZ_I = Y_I
        g = 1j * Y_II
        inc_term = 1j * n_I * np.cos(theta) * delta_i0

        # oneover_E_conv_all = np.zeros(len(E_conv_all))  # Dummy for TE case

    elif polarization == 1:  # TM
        Z_I = np.diag(k_I_z / (k0 * n_I ** 2))
        Z_II = np.diag(k_II_z / (k0 * n_II ** 2))

        YZ_I = Z_I
        g = 1j * Z_II
        inc_term = 1j * delta_i0 * np.cos(theta) / n_I

        # oneover_E_conv_all = permittivity_mapping(patterns, wl, period, fourier_order, oneover=True)

    else:
        raise ValueError

    T = np.eye(2 * fourier_order + 1)

    return f, YZ_I, g, inc_term, T


def SMM1(Kx, k0, n_I, n_II):
    # scattering matrix needed for 'gap medium'
    # if calculations shift with changing selection of gap media, this is BAD; it should not shift with choice of gap
    Wg, Vg, Kzg = hl.homogeneous_1D(Kx, k0, 1)
    # reflection medium



    Wr, Vr, Kzr = hl.homogeneous_1D(Kx, k0, n_I, tt=polarization)
    # transmission medium;
    Wt, Vt, Kzt = hl.homogeneous_1D(Kx, k0, n_II, tt=polarization)
    # S matrices for the reflection region
    Ar, Br = sm.A_B_matrices_half_space(Wr, Wg, Vr, Vg)  # make sure this order is right
    _, Sg = sm.S_R(Ar, Br)  # scatter matrix for the reflection region

    return Wg, Vg, Kzg, Wr, Vr, Kzr, Wt, Vt, Kzt, Ar, Br, Sg


def TMM2(k0, q, d, W, V, f, g, fourier_order, T):
    X = np.diag(np.exp(-k0 * q * d))

    W_i = np.linalg.inv(W)
    V_i = np.linalg.inv(V)

    a = 0.5 * (W_i @ f + V_i @ g)
    b = 0.5 * (W_i @ f - V_i @ g)

    a_i = np.linalg.inv(a)

    f = W @ (np.eye(2 * fourier_order + 1) + X @ b @ a_i @ X)
    g = V @ (np.eye(2 * fourier_order + 1) - X @ b @ a_i @ X)
    T = T @ a_i @ X

    return f, g, T


def SMM2(W, Wg, V, Vg, d, k0, Q, Sg):
    # calculating A and B matrices for scattering matrix
    # define S matrix for the GRATING REGION
    A, B = sm.A_B_matrices(W, Wg, V, Vg)
    _, S_dict = sm.S_layer(A, B, d, k0, Q)
    _, Sg = rs.RedhefferStar(Sg, S_dict)

    return A, B, S_dict, Sg


def TMM3(g, YZ_I, f, delta_i0, inc_term, T, k_I_z, k0, n_I, theta, polarization, k_II_z):
    Tl = np.linalg.inv(g + 1j * YZ_I @ f) @ (1j * YZ_I @ delta_i0 + inc_term)
    R = f @ Tl - delta_i0
    T = T @ Tl

    de_ri = R * np.conj(R) * np.real(k_I_z / (k0 * n_I * np.cos(theta)))
    if polarization == 0:
        de_ti = T * np.conj(T) * np.real(k_II_z / (k0 * n_I * np.cos(theta)))
    elif polarization == 1:
        de_ti = T * np.conj(T) * np.real(k_II_z / n_II ** 2) / (k0 * np.cos(theta) / n_I)
    else:
        raise ValueError

    return de_ri, de_ti


def SMM3(Wt, Wg, Vt, Vg, Sg, ff, Wr, fourier_order, Kzr, k0, k_I_z, Kzt, k_II_z):

    # define S matrices for the Transmission region
    At, Bt = sm.A_B_matrices_half_space(Wt, Wg, Vt, Vg)  # make sure this order is right
    _, St_dict = sm.S_T(At, Bt)  # scatter matrix for the reflection region
    _, Sg = rs.RedhefferStar(Sg, St_dict)

    # cinc is the incidence vector
    cinc = np.zeros((ff, 1))  # only need one set...
    cinc[fourier_order, 0] = 1
    cinc = np.linalg.inv(Wr) @ cinc
    # COMPUTE FIELDS: similar idea but more complex for RCWA since you have individual modes each contributing
    R = Wr @ Sg['S11'] @ cinc
    T = Wt @ Sg['S21'] @ cinc

    R, T, Kzr, Kzt = R.flatten(), T.flatten(), Kzr.T, Kzt.T

    if polarization == 0:
        de_ri = R * np.conj(R) @ np.real(Kzr / (n_I * np.cos(theta)))
        de_ti = T * np.conj(T) @ np.real(Kzt / (n_I * np.cos(theta)))
    elif polarization == 1:
        de_ri = R * np.conj(R) @ np.real(Kzr / (n_I * np.cos(theta))) / n_I ** 2
        de_ti = T * np.conj(T) @ np.real(Kzt / n_II ** 2) / (np.cos(theta) / n_I) / n_II ** 2
    else:
        raise ValueError

    return de_ri, de_ti


if __name__ == '__main__':
    n_I = 3
    n_II = 10

    theta = 0
    phi = 0
    psi = 0

    fourier_order = 10
    period = [0.7]
    period = [700]

    # wls = np.linspace(0.5, 2.3, 400)
    wls = np.linspace(500, 2300, 400)

    polarization = 0  # TE 0, TM 1

    # permittivity in grating layer
    # patterns = [[3.48, 1, 0.3], [3.48, 1, 0.3]]  # n_ridge, n_groove, fill_factor
    patterns = [['SILICON', 1, np.array([1, 1, 1, -1, -1, -1, -1, -1, -1, -1])],
                ['SILICON', 1, np.array([1, 1, 1, -1, -1, -1, -1, -1, -1, -1])]]  # n_ridge, n_groove, fill_factor

    thickness = [0.46, 0.66]
    thickness = [460, 660]
    # thickness = [325]

    polarization_type = 0

    res = RcwaBackbone(polarization_type, n_I, n_II, theta, phi, psi, fourier_order, period, wls,
                       polarization, patterns, thickness, algo='TMM')
    res.lalanne_1d()

    plt.plot(res.wls, res.spectrum_r.sum(axis=1))
    plt.plot(res.wls, res.spectrum_t.sum(axis=1))
    plt.show()


    res = RcwaBackbone(polarization_type, n_I, n_II, theta, phi, psi, fourier_order, period, wls,
                       polarization, patterns, thickness, algo='SMM')

    res.lalanne_1d()
    # res.lalanne_1d_conical()
    # res.lalanne_2d()

    plt.plot(res.wls, res.spectrum_r.sum(axis=1))
    plt.plot(res.wls, res.spectrum_t.sum(axis=1))
    plt.show()
