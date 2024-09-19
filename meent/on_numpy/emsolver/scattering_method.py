"""
currently SMM is not supported
"""

# many codes for scattering matrix method are from here:
# https://github.com/zhaonat/Rigorous-Coupled-Wave-Analysis
# also refer our fork https://github.com/yonghakim/zhaonat-rcwa

from .smm_util import *


def scattering_1d_1(k0, n_I, n_II, theta, phi, fourier_indices, period, pol, wl=None):

    kx_vector = (n_I * np.sin(theta) * np.cos(phi) - fourier_indices * (
                2 * np.pi / k0 / period[0])).astype('complex')
    Kx = np.diag(kx_vector)

    # scattering matrix needed for 'gap medium'
    Wg, Vg, Kzg = homogeneous_1D(Kx, 1, wl=wl, comment='Gap')

    # reflection medium
    Wr, Vr, Kzr = homogeneous_1D(Kx, n_I, pol=pol, wl=wl, comment='Refl')

    # transmission medium;
    Wt, Vt, Kzt = homogeneous_1D(Kx, n_II, pol=pol, wl=wl, comment='Tran')

    # S matrices for the reflection region
    Ar, Br = A_B_matrices_half_space(Vr, Vg)  # make sure this order is right
    _, Sg = S_RT(Ar, Br, ref_mode=True)  # scatter matrix for the reflection region

    return Kx, Wg, Vg, Kzg, Wr, Vr, Kzr, Wt, Vt, Kzt, Ar, Br, Sg


def scattering_1d_2(W, Wg, V, Vg, d, k0, LAMBDA, Sg):
    # calculating A and B matrices for scattering matrix
    # define S matrix for the GRATING REGION
    A, B = A_B_matrices(W, Wg, V, Vg)
    _, S_dict = S_layer(A, B, d, k0, LAMBDA)
    _, Sg = RedhefferStar(Sg, S_dict)

    return A, B, S_dict, Sg


def scattering_1d_3(Wt, Wg, Vt, Vg, Sg, ff, Wr, fourier_order, Kzr, Kzt, n_I, n_II, theta, pol):
    # define S matrices for the Transmission region
    At, Bt = A_B_matrices_half_space(Vt, Vg)  # make sure this order is right
    _, St_dict = S_RT(At, Bt, ref_mode=False)  # scatter matrix for the reflection region
    _, Sg = RedhefferStar(Sg, St_dict)

    k_inc = n_I * np.array([np.sin(theta), 0, np.cos(theta)])

    c_inc = np.zeros((ff, 1))  # only need one set...
    c_inc[fourier_order] = 1
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


def scattering_2d_1(n_I, n_II, theta, phi, k0, period, fourier_order, kx, ky):
    kx_inc = n_I * np.sin(theta) * np.cos(phi)
    ky_inc = n_I * np.sin(theta) * np.sin(phi)
    kz_inc = np.sqrt(n_I ** 2 * 1 - kx_inc ** 2 - ky_inc ** 2)

    Kx, Ky = K_matrix_cubic_2D(kx_inc, ky_inc, k0, period[0], period[1], fourier_order[0], fourier_order[1])
    print(Kx.shape, Ky.shape)
    # specify gap media (this is an LHI so no eigenvalue problem should be solved
    e_h = 1
    Wg, Vg, Kzg = homogeneous_module(Kx, Ky, e_h)

    # ================= Working on the Reflection Side =========== ##
    e_r = n_I ** 2
    Wr, Vr, Kzr = homogeneous_module(Kx, Ky, e_r)

    # ========= Working on the Transmission Side==============##
    e_t = n_II ** 2
    Wt, Vt, Kzt = homogeneous_module(Kx, Ky, e_t)

    # calculating A and B matrices for scattering matrix
    Ar, Br = A_B_matrices_half_space(Vr, Vg)

    # s_ref is a matrix, Sr_dict is a dictionary
    _, Sr_dict = S_RT(Ar, Br, ref_mode=True)  # scatter matrix for the reflection region
    Sg = Sr_dict

    ff_x, ff_y = fourier_order
    ff_xy = ff_x * ff_y

    # I = np.eye(ff_xy, dtype=type_complex)
    # O = np.zeros((ff_xy, ff_xy), dtype=type_complex)
    I = np.eye(ff_xy)
    O = np.zeros((ff_xy, ff_xy))

    # kz_top = (n_I ** 2 - Kx.diagonal() ** 2 - Ky.diagonal().reshape((-1, 1)) ** 2) ** 0.5
    # kz_bot = (n_II ** 2 - Kx.diagonal() ** 2 - Ky.diagonal().reshape((-1, 1)) ** 2) ** 0.5

    kz_top = (n_I ** 2 - kx ** 2 - ky.reshape((-1, 1)) ** 2) ** 0.5
    kz_bot = (n_II ** 2 - kx ** 2 - ky.reshape((-1, 1)) ** 2) ** 0.5

    kz_top = kz_top.flatten().conjugate()
    kz_bot = kz_bot.flatten().conjugate()

    return Kx, Ky, kz_inc, Wg, Vg, Kzg, Wr, Vr, Kzr, Wt, Vt, Kzt, Ar, Br, Sg, kz_top, kz_bot


def scattering_2d_2(W, Wg, V, Vg, d, k0, Sg, LAMBDA):

    A, B = A_B_matrices(W, Wg, V, Vg)
    _, Sl_dict = S_layer(A, B, d, k0, LAMBDA)
    Sg_matrix, Sg = RedhefferStar(Sg, Sl_dict)

    return A, B, Sl_dict, Sg_matrix, Sg


def scattering_2d_3(Wt, Wg, Vt, Vg, Sg, Wr, Kx, Ky, Kzr, Kzt, kz_top, kz_bot, n_top, n_bot, pol, theta,
                    phi, fourier_order):
    normal_vector = np.array([0, 0, 1])  # positive z points down;
    # amplitude of the te vs tm modes (which are decoupled)

    if pol == 0:
        pte = 1
        ptm = 0
    elif pol == 1:
        pte = 0
        ptm = 1
    else:
        raise ValueError

    M, N = fourier_order
    # NM = ff ** 2
    NM = ((2*M+1)*(2*N+1))

    # get At, Bt
    # since transmission is the same as gap, order does not matter
    At, Bt = A_B_matrices_half_space(Vt, Vg)
    _, ST_dict = S_RT(At, Bt, ref_mode=False)

    # update global scattering matrix
    Sg_matrix, Sg = RedhefferStar(Sg, ST_dict)

    # finally CONVERT THE GLOBAL SCATTERING MATRIX BACK TO A MATRIX

    K_inc_vector = n_top * np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])

    _, e_src, _ = initial_conditions(K_inc_vector, theta, normal_vector, pte, ptm, N, M)

    c_inc = np.linalg.inv(Wr) @ e_src
    # COMPUTE FIELDS: similar idea but more complex for RCWA since you have individual modes each contributing
    reflected = Wr @ Sg['S11'] @ c_inc
    transmitted = Wt @ Sg['S21'] @ c_inc

    R_s = np.array(reflected[0:NM, :]).flatten()  # rx is the Ex component.
    R_p = np.array(reflected[NM:, :]).flatten()
    T_s = np.array(transmitted[0:NM, :]).flatten()
    T_p = np.array(transmitted[NM:, :]).flatten()
    # R_s = reflected[0:NM, 0]  # rx is the Ex component.
    # R_p = reflected[NM:, 0]
    # T_s = transmitted[0:NM, 0]
    # T_p = transmitted[NM:, 0]

    # rz = np.linalg.inv(Kzr) @ (Kx @ R_s + Ky @ R_p)
    # tz = np.linalg.inv(Kzt) @ (Kx @ T_s + Ky @ T_p)
    #
    # rsq = np.square(np.abs(R_s)) + np.square(np.abs(R_p)) + np.square(np.abs(rz))
    # tsq = np.square(np.abs(T_s)) + np.square(np.abs(T_p)) + np.square(np.abs(tz))
    #
    # de_ri = np.real(Kzr)@rsq/np.real(K_inc_vector[2])  # real because we only want propagating components
    # de_ti = np.real(Kzt)@tsq/np.real(K_inc_vector[2])

    # return de_ri, de_ti

    print(R_s.shape, kz_top.shape)
    de_ri_s = R_s * np.conj(R_s) * np.real(kz_top / (n_top * np.cos(theta)))
    de_ri_p = R_p * np.conj(R_p) * np.real(kz_top / n_top ** 2 / (n_top * np.cos(theta)))

    de_ti_s = T_s * np.conj(T_s) * np.real(kz_bot / (n_top * np.cos(theta)))
    de_ti_p = T_p * np.conj(T_p) * np.real(kz_bot / n_bot ** 2 / (n_top * np.cos(theta)))


    # return de_ri.real, de_ti.real, big_T1
    return de_ri_s.real, de_ri_p.real, de_ti_s.real, de_ti_p.real, R_s, R_p, T_s, T_p

def scattering_2d_wv(Kx, Ky, epx_conv, epy_conv, epz_conv_i, mu_conv=None):
    # -------------------------
    # W and V from SMM method.
    # NM = ff ** 2
    # M, N = fourier_order
    # print(N,M)
    # NM = ((2*M+1)*(2*N+1))
    NM = len(Kx)
    if mu_conv is None:
        mu_conv = np.identity(NM)

    P, Q, _ = P_Q_kz(Kx, Ky, epx_conv, epy_conv, epz_conv_i, mu_conv)
    GAMMA = P @ Q

    Lambda, W = np.linalg.eig(GAMMA)  # LAMBDa is effectively refractive index
    LAMBDA = np.diag(Lambda)
    LAMBDA = np.sqrt(LAMBDA.astype('complex'))

    V = Q @ W @ np.linalg.inv(LAMBDA)

    return W, V, LAMBDA
