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


def scattering_1d_1(k0, n_I, n_II, theta, phi, fourier_indices, period, pol, wl=None):

    kx_vector = (n_I * np.sin(theta) * np.cos(phi) - fourier_indices * (
                2 * np.pi / k0 / period[0])).astype('complex')
    Kx = np.diag(kx_vector)

    # scattering matrix needed for 'gap medium'
    # if calculations shift with changing selection of gap media, this is BAD; it should not shift with choice of gap

    Wg, Vg, Kzg = hl.homogeneous_1D(Kx, 1, wl=wl, comment='Gap')

    # reflection medium
    Wr, Vr, Kzr = hl.homogeneous_1D(Kx, n_I, pol=pol, wl=wl, comment='Refl')

    # transmission medium;
    Wt, Vt, Kzt = hl.homogeneous_1D(Kx, n_II, pol=pol, wl=wl, comment='Tran')


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


def scattering_1d_3(Wt, Wg, Vt, Vg, Sg, ff, Wr, fourier_order, Kzr, Kzt, n_I, n_II, theta, pol, Kx):
    # define S matrices for the Transmission region
    At, Bt = sm.A_B_matrices_half_space(Wt, Wg, Vt, Vg)  # make sure this order is right
    _, St_dict = sm.S_T(At, Bt)  # scatter matrix for the reflection region
    _, Sg = rs.RedhefferStar(Sg, St_dict)

    # c_inc is the incidence vector
    delta = np.zeros((ff, 1))  # only need one set...
    delta[fourier_order, 0] = 1

    k_inc = n_I * np.array([np.sin(theta), 0, np.cos(theta)])

    c_inc = np.zeros((ff, ))  # only need one set...
    c_inc[fourier_order] = 1
    c_inc = c_inc.T
    c_inc = np.linalg.inv(Wr) @ c_inc
    # COMPUTE FIELDS: similar idea but more complex for RCWA since you have individual modes each contributing
    reflected = Wr @ Sg['S11'] @ c_inc
    transmitted = Wt @ Sg['S21'] @ c_inc

    # reflected is already ry or Ey
    rsq = np.square(np.abs(reflected))
    tsq = np.square(np.abs(transmitted))

    # compute final reflectivity
    if pol == 0:
        de_ri = np.real(Kzr) @ rsq / np.real(k_inc[2])
        de_ti = np.real(Kzt) @ tsq / np.real(k_inc[2])
    elif pol == 1:
        de_ri = np.real(Kzr)@rsq/np.real(k_inc[2]) / n_I**2
        de_ti = np.real(Kzt)@tsq/np.real(k_inc[2]) * n_I**2 / n_II**4
    else:
        raise ValueError

    return de_ri.flatten(), de_ti.flatten()


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

    rsq = np.square(np.abs(rx)) + np.square(np.abs(ry)) + np.square(np.abs(rz))
    tsq = np.square(np.abs(tx)) + np.square(np.abs(ty)) + np.square(np.abs(tz))

    de_ri = np.real(Kzr) @ rsq / np.real(kz_inc)
    de_ti = np.real(Kzt) @ tsq / np.real(kz_inc)

    # compute final reflectivity

    # de_ri = np.real(Kzr)@rsq/np.real(K_inc_vector[2])  # real because we only want propagating components
    # de_ti = np.real(Kzt)@tsq/np.real(K_inc_vector[2])


    # print(de_ri.sum(), de_ti.sum())

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
