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


def scattering_1d_1(k0, n_I, n_II, theta, phi, fourier_indices, period, pol):

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


# def scattering_1d_1(k0, n_I, n_II, theta, phi, fourier_indices, period, pol):
#
#     kx_inc = n_I * np.sin(theta)
#     kz_inc = np.sqrt(n_I ** 2 * 1 - kx_inc ** 2)
#     kz_inc = np.conj(kz_inc)
#
#     kx_vector = (n_I * np.sin(theta) * np.cos(phi) - fourier_indices * (
#                 2 * np.pi / k0 / period[0])).astype('complex')
#     Kx = np.diag(kx_vector)
#
#     # scattering matrix needed for 'gap medium'
#     # if calculations shift with changing selection of gap media, this is BAD; it should not shift with choice of gap
#
#     # Wg, Vg, Kzg = hl.homogeneous_1D(Kx, 1)
#
#     I = np.identity(len(Kx))
#
#     W_g = I
#     Q_g = I - Kx ** 2
#
#     Kz_g = np.sqrt(I - Kx ** 2)
#     Kz_g = np.conj(Kz_g)  # TODO: conjugate?
#
#     eigenvalues_g = 1j * Kz_g  # determining the modes of ex, ey... so it appears eigenvalue order MATTERS...
#     V_g = Q_g @ np.linalg.inv(eigenvalues_g)  # eigenvalue order is arbitrary (hard to compare with matlab
#
#
#     # reflection medium
#     # Wr, Vr, Kzr = hl.homogeneous_1D(Kx, n_I, pol=pol)
#
#     W_r = I
#     Q_r = n_I**2 * I - Kx ** 2
#
#     Kz_r = -np.conj(np.sqrt(n_I**2 * I - Kx**2))
#
#     # Kz_r = np.sqrt(I - Kx ** 2)
#     # Kz_r = np.conj(Kz_r)  # TODO: conjugate?
#
#     eigenvalues_r = -1j * Kz_r  # determining the modes of ex, ey... so it appears eigenvalue order MATTERS...
#     V_r = Q_r @ np.linalg.inv(eigenvalues_r)  # eigenvalue order is arbitrary (hard to compare with matlab
#
#     # transmission medium;
#     # Wt, Vt, Kzt = hl.homogeneous_1D(Kx, n_II, pol=pol)
#
#     W_t = I
#     Q_t = n_II**2 * I - Kx ** 2
#
#     Kz_t = np.conj(np.sqrt(n_II**2 * I - Kx**2))
#
#     eigenvalues_t = 1j * Kz_t  # determining the modes of ex, ey... so it appears eigenvalue order MATTERS...
#     V_t = Q_t @ np.linalg.inv(eigenvalues_t)  # eigenvalue order is arbitrary (hard to compare with matlab
#
#     Wr, Vr, Kzr = W_r, V_r, Kz_r
#     Wg, Vg, Kzg = W_g, V_g, Kz_g
#     Wt, Vt, Kzt = W_t, V_t, Kz_t
#
#
#     # S matrices for the reflection region
#     Ar, Br = sm.A_B_matrices_half_space(Wr, Wg, Vr, Vg)  # make sure this order is right
#     _, Sg = sm.S_R(Ar, Br)  # scatter matrix for the reflection region
#
#     return Kx, Wg, Vg, Kzg, Wr, Vr, Kzr, Wt, Vt, Kzt, Ar, Br, Sg


def scattering_1d_2(W, Wg, V, Vg, d, k0, LAMBDA, Sg):
    # calculating A and B matrices for scattering matrix
    # define S matrix for the GRATING REGION
    A, B = sm.A_B_matrices(W, Wg, V, Vg)
    _, S_dict = sm.S_layer(A, B, d, k0, LAMBDA)
    _, Sg = rs.RedhefferStar(Sg, S_dict)

    return A, B, S_dict, Sg


def scattering_1d_3(Wt, Wg, Vt, Vg, Sg, ff, Wr, fourier_order, Kzr, Kzt, n_I, n_II, theta, pol, Kx):
    # define S matrices for the Transmission region
    At, Bt = sm.A_B_matrices_half_space(Wt, Wg, Vt, Vg)  # make sure this order is right
    _, St_dict = sm.S_T(At, Bt)  # scatter matrix for the reflection region
    _, Sg = rs.RedhefferStar(Sg, St_dict)

    # cinc is the incidence vector
    delta = np.zeros((ff, 1))  # only need one set...
    delta[fourier_order, 0] = 1

    k_inc = n_I * np.array([np.sin(theta), 0, np.cos(theta)])

    n_vector = np.array([0, 0, 1])
    a_te = np.cross(n_vector, k_inc)
    a_te = a_te / np.linalg.norm(a_te)

    a_tm = np.cross(k_inc, a_te)
    a_tm = a_tm / np.linalg.norm(a_tm)

    p_te = 0
    p_tm = 1

    P = p_te * a_te + p_tm * a_tm
    P = P / np.linalg.norm(P)

    e_t_src = P[0] * delta

    c_src = np.linalg.inv(Wr) @ e_t_src

    c_ref = Sg['S11'] @ c_src
    c_trn = Sg['S21'] @ c_src

    rx = e_t_ref = Wr @ c_ref
    tx = e_t_trn = Wt @ c_trn

    rz = - np.linalg.inv(Kzr) @ (Kx @ rx)
    tz = - np.linalg.inv(Kzt) @ (Kx @ tx)

    r2 = np.abs(rx)**2 + np.abs(rz)**2
    t2 = np.abs(tx)**2 + np.abs(tz)**2

    # kx_inc = n_I * np.sin(theta)
    # kz_inc = np.sqrt(n_I ** 2 * 1 - kx_inc ** 2)
    # kz_inc = np.conj(kz_inc)  # TODO: Check

    de_ri = np.real(-Kzr) / np.real(k_inc[2]) @ r2
    de_ti = np.real(Kzt) / np.real(k_inc[2]) @ t2

    # COMPUTE FIELDS: similar idea but more complex for RCWA since you have individual modes each contributing
    # R_x = Wr @ Sg['S11'] @ cinc
    # T_x = Wt @ Sg['S21'] @ cinc
    #
    #
    # R2 = np.abs(R_x)**2 + np.abs(R_z)**2
    # T2 = np.abs(T_x)**2 + np.abs(T_z)**2
    #
    # R2 = R_x * np.conj(R_x) + R_z * np.conj(R_z)
    # T2 = T_x * np.conj(T_x) + T_z * np.conj(T_z)
    #
    # # R, T, Kzr, Kzt = R.flatten(), T.flatten(), Kzr.T, Kzt.T
    # # R2, T2, Kzr, Kzt = R2.flatten(), T2.flatten(), Kzr.T, Kzt.T
    #
    if pol == 0:
        # de_ri = R * np.conj(R) @ np.real(Kzr / (n_I * np.cos(theta)))
        # de_ti = T * np.conj(T) @ np.real(Kzt / (n_I * np.cos(theta)))
        de_ri = np.real(-Kzr) / np.real(k_inc[2]) @ r2 /(n_I * np.cos(theta))
        de_ti = np.real(Kzt) / np.real(k_inc[2]) @ t2 /(n_I * np.cos(theta))

    elif pol == 1:
        # de_ri = R * np.conj(R) @ np.real(Kzr / (n_I * np.cos(theta))) / n_I ** 2
        # de_ti = T * np.conj(T) @ np.real(Kzt / n_II ** 2) / (np.cos(theta) / n_I) / n_II ** 2
        de_ri = np.real(Kzr) / np.real(k_inc[2]) @ r2 / (n_I * np.cos(theta)) / n_I ** 2
        de_ti = np.real(Kzt) / np.real(k_inc[2]) @ t2 / (np.cos(theta) / n_I) / n_II ** 2

    # #
    # # else:
    # #     raise ValueError
    #
    #
    # kx_inc = n_I * np.sin(theta)
    # # ky_inc = n1 * np.sin(self.theta) * np.sin(self.phi)
    # kz_inc = np.sqrt(n_I ** 2 * 1 - kx_inc ** 2)
    # kz_inc = np.conj(kz_inc)  # TODO: Check
    #
    # # de_ri = R * np.conj(R) @ np.real(-Kzr / kz_inc)
    # # de_ti = T * np.conj(T) @ np.real(Kzt / kz_inc)
    #
    # # de_ri = R2 @ np.real(-Kzr / kz_inc)
    # # de_ti = T2 @ np.real(Kzt / kz_inc)
    #
    # de_ri = np.real(-Kzr / kz_inc) @ R2
    # de_ti = np.real(Kzt / kz_inc) @ T2

    cinc = np.zeros((ff, ))  # only need one set...
    cinc[fourier_order] = 1
    cinc = cinc.T
    cinc = np.linalg.inv(Wr) @ cinc
    # COMPUTE FIELDS: similar idea but more complex for RCWA since you have individual modes each contributing
    reflected = Wr @ Sg['S11'] @ cinc
    transmitted = Wt @ Sg['S21'] @ cinc

    # reflected is already ry or Ey
    rsq = np.square(np.abs(reflected))
    tsq = np.square(np.abs(transmitted))

    # compute final reflectivity
    de_ri = np.real(Kzr)@rsq/np.real(k_inc[2]) / n_I**2  # real because we only want propagating components
    de_ti = np.real(Kzt)@tsq/np.real(k_inc[2]) * n_I**2 / n_II**4

    return de_ri.flatten(), de_ti.flatten()

# def scattering_1d_3(Wt, Wg, Vt, Vg, Sg, ff, Wr, fourier_order, Kzr, Kzt, n_I, n_II, theta, pol, Kx):
#     # define S matrices for the Transmission region
#     At, Bt = sm.A_B_matrices_half_space(Wt, Wg, Vt, Vg)  # make sure this order is right
#     _, St_dict = sm.S_T(At, Bt)  # scatter matrix for the reflection region
#     _, Sg = rs.RedhefferStar(Sg, St_dict)
#
#     # cinc is the incidence vector
#     delta = np.zeros((ff, 1))  # only need one set...
#     delta[fourier_order, 0] = 1
#
#     k_inc = n_I*np.array([np.sin(theta), 0, np.cos(theta)])
#
#     n_vector = np.array([0, 0, 1])
#     a_te = np.cross(n_vector, k_inc)
#     a_te = a_te / np.linalg.norm(a_te)
#
#     a_tm = np.cross(k_inc, a_te)
#     a_tm = a_tm / np.linalg.norm(a_tm)
#
#     p_te = 0
#     p_tm = 1
#
#     P = p_te * a_te + p_tm * a_tm
#     P = P / np.linalg.norm(P)
#
#     e_t_src = P[0] * delta
#
#     c_src = np.linalg.inv(Wr) @ e_t_src
#
#     c_ref = Sg['S11'] @ c_src
#     c_trn = Sg['S21'] @ c_src
#
#     rx = e_t_ref = Wr @ c_ref
#     tx = e_t_trn = Wt @ c_trn
#
#     rz = - np.linalg.inv(Kzr) @ (Kx @ rx)
#     tz = - np.linalg.inv(Kzt) @ (Kx @ tx)
#
#     r2 = np.abs(rx)**2 + np.abs(rz)**2
#     t2 = np.abs(tx)**2 + np.abs(tz)**2
#
#     # kx_inc = n_I * np.sin(theta)
#     # kz_inc = np.sqrt(n_I ** 2 * 1 - kx_inc ** 2)
#     # kz_inc = np.conj(kz_inc)  # TODO: Check
#
#     de_ri = np.real(-Kzr) / np.real(k_inc[2]) @ r2
#     de_ti = np.real(Kzt) / np.real(k_inc[2]) @ t2
#
#     # COMPUTE FIELDS: similar idea but more complex for RCWA since you have individual modes each contributing
#     # R_x = Wr @ Sg['S11'] @ cinc
#     # T_x = Wt @ Sg['S21'] @ cinc
#     #
#     #
#     # R2 = np.abs(R_x)**2 + np.abs(R_z)**2
#     # T2 = np.abs(T_x)**2 + np.abs(T_z)**2
#     #
#     # R2 = R_x * np.conj(R_x) + R_z * np.conj(R_z)
#     # T2 = T_x * np.conj(T_x) + T_z * np.conj(T_z)
#     #
#     # # R, T, Kzr, Kzt = R.flatten(), T.flatten(), Kzr.T, Kzt.T
#     # # R2, T2, Kzr, Kzt = R2.flatten(), T2.flatten(), Kzr.T, Kzt.T
#     #
#     # # if pol == 0:
#     # #     de_ri = R * np.conj(R) @ np.real(Kzr / (n_I * np.cos(theta)))
#     # #     de_ti = T * np.conj(T) @ np.real(Kzt / (n_I * np.cos(theta)))
#     # # elif pol == 1:
#     # #     de_ri = R * np.conj(R) @ np.real(Kzr / (n_I * np.cos(theta))) / n_I ** 2
#     # #     de_ti = T * np.conj(T) @ np.real(Kzt / n_II ** 2) / (np.cos(theta) / n_I) / n_II ** 2
#     # #
#     # # else:
#     # #     raise ValueError
#     #
#     #
#     # kx_inc = n_I * np.sin(theta)
#     # # ky_inc = n1 * np.sin(self.theta) * np.sin(self.phi)
#     # kz_inc = np.sqrt(n_I ** 2 * 1 - kx_inc ** 2)
#     # kz_inc = np.conj(kz_inc)  # TODO: Check
#     #
#     # # de_ri = R * np.conj(R) @ np.real(-Kzr / kz_inc)
#     # # de_ti = T * np.conj(T) @ np.real(Kzt / kz_inc)
#     #
#     # # de_ri = R2 @ np.real(-Kzr / kz_inc)
#     # # de_ti = T2 @ np.real(Kzt / kz_inc)
#     #
#     # de_ri = np.real(-Kzr / kz_inc) @ R2
#     # de_ti = np.real(Kzt / kz_inc) @ T2
#
#     return de_ri.flatten(), de_ti.flatten()


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
