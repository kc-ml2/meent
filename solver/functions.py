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


def transfer_1d_1(ff, polarization, k0, n_I, n_II, theta, delta_i0, fourier_order,fourier_indices, wl, period):

    kx_vector = k0 * (n_I * np.sin(theta) - fourier_indices * (wl / period)).astype('complex')

    k_I_z = (k0 ** 2 * n_I ** 2 - kx_vector ** 2) ** 0.5
    k_II_z = (k0 ** 2 * n_II ** 2 - kx_vector ** 2) ** 0.5

    k_I_z = k_I_z.conjugate()
    k_II_z = k_II_z.conjugate()

    Kx = np.diag(kx_vector / k0)

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

    return Kx, k_I_z, k_II_z, Kx, f, YZ_I, g, inc_term, T


def transfer_1d_2(k0, q, d, W, V, f, g, fourier_order, T):
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


def transfer_1d_3(g, YZ_I, f, delta_i0, inc_term, T, k_I_z, k0, n_I, n_II, theta, polarization, k_II_z):
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


def transfer_2d_1(ff, k0, n_I, n_II, period, fourier_indices, theta, phi, wl):
    I = np.eye(ff ** 2)
    O = np.zeros((ff ** 2, ff ** 2))

    kx_vector = k0 * (n_I * np.sin(theta) * np.cos(phi) - fourier_indices * (
            wl / period[0])).astype('complex')
    ky_vector = k0 * (n_I * np.sin(theta) * np.sin(phi) - fourier_indices * (
            wl / period[1])).astype('complex')

    Kx = np.diag(np.tile(kx_vector, ff).flatten()) / k0
    Ky = np.diag(np.tile(ky_vector.reshape((-1, 1)), ff).flatten()) / k0

    k_I_z = (k0 ** 2 * n_I ** 2 - kx_vector ** 2 - ky_vector.reshape((-1, 1)) ** 2) ** 0.5
    k_II_z = (k0 ** 2 * n_II ** 2 - kx_vector ** 2 - ky_vector.reshape((-1, 1)) ** 2) ** 0.5

    k_I_z = k_I_z.flatten().conjugate()
    k_II_z = k_II_z.flatten().conjugate()

    varphi = np.arctan(ky_vector.reshape((-1, 1)) / kx_vector).flatten()

    Y_I = np.diag(k_I_z / k0)
    Y_II = np.diag(k_II_z / k0)

    Z_I = np.diag(k_I_z / (k0 * n_I ** 2))
    Z_II = np.diag(k_II_z / (k0 * n_II ** 2))

    big_F = np.block([[I, O], [O, 1j * Z_II]])
    big_G = np.block([[1j * Y_II, O], [O, I]])

    big_T = np.eye(ff ** 2 * 2)

    return Kx, Ky, k_I_z, k_II_z, varphi, Y_I, Y_II, Z_I, Z_II, big_F, big_G, big_T


def transfer_2d_2(k0, d, W, V, center, Lambda, varphi, I, O, big_F, big_G, big_T):

    Lambda_1 = Lambda[:center]
    Lambda_2 = Lambda[center:]

    W_11 = W[:center, :center]
    W_12 = W[:center, center:]
    W_21 = W[center:, :center]
    W_22 = W[center:, center:]

    V_11 = V[:center, :center]
    V_12 = V[:center, center:]
    V_21 = V[center:, :center]
    V_22 = V[center:, center:]

    X_1 = np.diag(np.exp(-k0 * Lambda_1 * d))
    X_2 = np.diag(np.exp(-k0 * Lambda_2 * d))

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

    return big_F, big_G, big_T


def transfer_2d_3(center, big_F, big_G, big_T, Z_I, Y_I, psi, theta, ff, delta_i0, k_I_z, k0, n_I, n_II, k_II_z):
    I = np.eye(ff ** 2)
    O = np.zeros((ff ** 2, ff ** 2))

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

    final_B = np.block(
        [
            [-np.sin(psi) * delta_i0],
            [-np.cos(psi) * np.cos(theta) * delta_i0],
            [-1j * np.sin(psi) * n_I * np.cos(theta) * delta_i0],
            [1j * n_I * np.cos(psi) * delta_i0]
        ]
    )

    final_X = np.linalg.inv(final_A) @ final_B

    R_s = final_X[:ff ** 2, :].flatten()
    R_p = final_X[ff ** 2:2 * ff ** 2, :].flatten()

    big_T = big_T @ final_X[2 * ff ** 2:, :]
    T_s = big_T[:ff ** 2, :].flatten()
    T_p = big_T[ff ** 2:, :].flatten()

    de_ri = R_s * np.conj(R_s) * np.real(k_I_z / (k0 * n_I * np.cos(theta))) \
            + R_p * np.conj(R_p) * np.real((k_I_z / n_I ** 2) / (k0 * n_I * np.cos(theta)))

    de_ti = T_s * np.conj(T_s) * np.real(k_II_z / (k0 * n_I * np.cos(theta))) \
            + T_p * np.conj(T_p) * np.real((k_II_z / n_II ** 2) / (k0 * n_I * np.cos(theta)))

    Aa = de_ri.sum()
    Aaa = de_ti.sum()

    if Aa + Aaa != 1:
        # TODO: no problem? or should be handled?
        print(1)
        wl = 1463.6363636363637
        deri = 350

        wl = 1978.9715332727274
        deri = 558

    return de_ri, de_ti


def scattering_1d_1(k0, n_I, n_II, theta, phi,fourier_indices, period, pol):
    kx_inc = n_I * np.sin(theta) * np.cos(phi)
    kz_inc = np.sqrt(n_I ** 2 * 1 - kx_inc ** 2)

    kx_vector = (n_I * np.sin(theta) * np.cos(phi) - fourier_indices * (
                2 * np.pi / k0 / period[0])).astype('complex')
    Kx = np.diag(kx_vector)

    # scattering matrix needed for 'gap medium'
    # if calculations shift with changing selection of gap media, this is BAD; it should not shift with choice of gap
    Wg, Vg, Kzg = hl.homogeneous_1D(Kx, 1)
    # reflection medium

    Wr, Vr, Kzr = hl.homogeneous_1D(Kx, n_I, pol=pol)
    # transmission medium;
    Wt, Vt, Kzt = hl.homogeneous_1D(Kx, n_II, pol=pol)
    # S matrices for the reflection region
    Ar, Br = sm.A_B_matrices_half_space(Wr, Wg, Vr, Vg)  # make sure this order is right
    _, Sg = sm.S_R(Ar, Br)  # scatter matrix for the reflection region

    return Kx, Wg, Vg, Kzg, Wr, Vr, Kzr, Wt, Vt, Kzt, Ar, Br, Sg


def scattering_1d_2(W, Wg, V, Vg, d, k0, LAMBDA, Sg):
    # calculating A and B matrices for scattering matrix
    # define S matrix for the GRATING REGION
    A, B = sm.A_B_matrices(W, Wg, V, Vg)
    _, S_dict = sm.S_layer(A, B, d, k0, LAMBDA)
    _, Sg = rs.RedhefferStar(Sg, S_dict)

    return A, B, S_dict, Sg


def scattering_1d_3(Wt, Wg, Vt, Vg, Sg, ff, Wr, fourier_order, Kzr, Kzt, n_I, n_II, theta, pol):
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

    # de_ri = R * np.conj(R) @ np.real(Kzr / (n_I * np.cos(theta)))
    # de_ti = T * np.conj(T) @ np.real(Kzt / (n_I * np.cos(theta)))

    if pol == 0:
        de_ri = R * np.conj(R) @ np.real(Kzr / (n_I * np.cos(theta)))
        de_ti = T * np.conj(T) @ np.real(Kzt / (n_I * np.cos(theta)))
    elif pol == 1:
        de_ri = R * np.conj(R) @ np.real(Kzr / (n_I * np.cos(theta))) / n_I ** 2
        de_ti = T * np.conj(T) @ np.real(Kzt / n_II ** 2) / (np.cos(theta) / n_I) / n_II ** 2
    else:
        raise ValueError

    return de_ri, de_ti


def scattering_2d_1(n_I, n_II, theta, phi, k0, period, fourier_order):
    kx_inc = n_I * np.sin(theta) * np.cos(phi)
    ky_inc = n_I * np.sin(theta) * np.sin(phi)
    kz_inc = np.sqrt(n_I ** 2 * 1 - kx_inc ** 2 - ky_inc ** 2)

    Kx, Ky = km.K_matrix_cubic_2D(kx_inc, ky_inc, k0, period[0], period[1], fourier_order,
                                    fourier_order);  # Kx and Ky are diagonal but have a 0 on it

    ## =============== K Matrices for gap medium =========================
    ## specify gap media (this is an LHI so no eigenvalue problem should be solved
    e_h = 1
    Wg, Vg, Kzg = hl.homogeneous_module(Kx, Ky, e_h)

    ### ================= Working on the Reflection Side =========== ##
    e_r = n_I ** 2
    Wr, Vr, Kzr = hl.homogeneous_module(Kx, Ky, e_r)

    ##========= Working on the Transmission Side==============##
    e_t = n_II ** 2
    Wt, Vt, Kzt = hl.homogeneous_module(Kx, Ky, e_t)

    ## calculating A and B matrices for scattering matrix
    # since gap medium and reflection media are the same, this doesn't affect anything
    Ar, Br = sm.A_B_matrices_half_space_new(Wr, Wg, Vr, Vg)  # TODO: half space?

    ## s_ref is a matrix, Sr_dict is a dictionary
    _, Sr_dict = sm.S_R(Ar, Br)  # scatter matrix for the reflection region
    # S_matrices.append(S_ref)
    Sg = Sr_dict

    return Kx, Ky, kz_inc, Wg, Vg, Kzg, Wr, Vr, Kzr, Wt, Vt, Kzt, Ar, Br, Sg


def scattering_2d_2(W, Wg, V, Vg, d, k0, Sg, LAMBDA):

    # now defIne A and B, slightly worse conditioned than W and V
    A, B = sm.A_B_matrices(W, Wg, V, Vg)  # ORDER HERE MATTERS A LOT because W_i is not diagonal

    # calculate scattering matrix
    # Li = layer_thicknesses[i];
    _, Sl_dict = sm.S_layer(A, B, d, k0, LAMBDA)
    # S_matrices.append(S_layer);

    # update global scattering matrix using redheffer star
    Sg_matrix, Sg = rs.RedhefferStar(Sg, Sl_dict)

    return A, B, Sl_dict, Sg_matrix, Sg


def scattering_2d_3(Wt, Wg, Vt, Vg, Sg, Wr, Kx, Ky, Kzr, Kzt, kz_inc, n_I, pol, theta,
                    phi, fourier_order, ff):
    normal_vector = np.array([0, 0, 1])  # positive z points down;
    # ampltidue of the te vs tm modes (which are decoupled)

    if pol == 0:
        pte = 1
        ptm = 0
    elif pol == 1:
        pte = 0
        ptm = 1
    else:
        raise ValueError

    M = N = fourier_order
    NM = ff ** 2

    # get At, Bt
    # since transmission is the same as gap, order does not matter
    At, Bt = sm.A_B_matrices_half_space_new(Wt, Wg, Vt, Vg)
    _, ST_dict = sm.S_T(At, Bt)

    # update global scattering matrix
    Sg_matrix, Sg = rs.RedhefferStar(Sg, ST_dict)

    # finally CONVERT THE GLOBAL SCATTERING MATRIX BACK TO A MATRIX

    K_inc_vector = n_I * np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])

    _, cinc, Polarization = ic.initial_conditions(K_inc_vector, theta, normal_vector, pte, ptm, N, M)

    cinc = np.linalg.inv(Wr) @ cinc

    # COMPUTE FIELDS: similar idea but more complex for RCWA since you have individual modes each contributing
    reflected = Wr @ Sg['S11'] @ cinc
    transmitted = Wt @ Sg['S21'] @ cinc

    rx = reflected[0:NM, :]  # rx is the Ex component.
    ry = reflected[NM:, :]
    tx = transmitted[0:NM, :]
    ty = transmitted[NM:, :]

    rz = np.linalg.inv(Kzr) @ (Kx @ rx + Ky @ ry)
    tz = np.linalg.inv(Kzt) @ (Kx @ tx + Ky @ ty)

    r_sq = np.square(np.abs(rx)) + np.square(np.abs(ry)) + np.square(np.abs(rz))
    t_sq = np.square(np.abs(tx)) + np.square(np.abs(ty)) + np.square(np.abs(tz))

    de_ri = np.real(Kzr) @ r_sq / np.real(kz_inc)
    de_ti = np.real(Kzt) @ t_sq / np.real(kz_inc)

    print(de_ri.sum(), de_ti.sum())

    return de_ri, de_ti


def transfer_2d_wv(ff, Kx, E_i, Ky, oneover_E_conv_i, E_conv, center):

    I = np.eye(ff ** 2)
    O = np.zeros((ff ** 2, ff ** 2))

    B = Kx @ E_i @ Kx - I
    D = Ky @ E_i @ Ky - I

    S2_from_S = np.block(
        [
            [Ky ** 2 + B @ oneover_E_conv_i, Kx @ (E_i @ Ky @ E_conv - Ky)],
            [Ky @ (E_i @ Kx @ oneover_E_conv_i - Kx), Kx ** 2 + D @ E_conv]
        ])

    # TODO: using eigh
    eigenvalues, W = np.linalg.eig(S2_from_S)

    Lambda = eigenvalues ** 0.5

    # Lambda_1 = Lambda[:center]
    # Lambda_2 = Lambda[center:]

    LAMBDA = np.diag(Lambda)
    LAMBDA_i = np.linalg.inv(LAMBDA)
    U1_from_S = np.block(
        [
            [-Kx @ Ky, Kx ** 2 - E_conv],
            [oneover_E_conv_i - Ky ** 2, Ky @ Kx]  # TODO Check x y order
        ]
    )
    V = U1_from_S @ W @ LAMBDA_i

    return W, V, LAMBDA, Lambda


def scattering_2d_wv(ff, Kx, Ky, E_conv, oneover_E_conv, oneover_E_conv_i, E_i, mu_conv=None):
    # -------------------------
    # W and V from SMM method.
    NM = ff ** 2
    if mu_conv is None:
        mu_conv = np.identity(NM)

    P, Q, _ = pq.P_Q_kz(Kx, Ky, E_conv, mu_conv, oneover_E_conv, oneover_E_conv_i, E_i)
    Gamma_squared = P @ Q

    W, LAMBDA = em.eigen_W(Gamma_squared)
    V = em.eigen_V(Q, W, LAMBDA)

    return W, V, LAMBDA
