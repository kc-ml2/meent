import numpy as np

from .primitives import meeinv


def transfer_1d_1(pol, kx, n_top, n_bot, type_complex=np.complex128):
    ff_x = len(kx)

    kz_top = (n_top ** 2 - kx ** 2) ** 0.5
    kz_bot = (n_bot ** 2 - kx ** 2) ** 0.5

    kz_top = kz_top.conj()
    kz_bot = kz_bot.conj()

    F = np.eye(ff_x, dtype=type_complex)

    if pol == 0:  # TE
        Kz_bot = np.diag(kz_bot)
        G = 1j * Kz_bot
    elif pol == 1:  # TM
        Kz_bot = np.diag(kz_bot / (n_bot ** 2))
        G = 1j * Kz_bot
    else:
        raise ValueError

    T = np.eye(ff_x, dtype=type_complex)

    return kz_top, kz_bot, F, G, T


def transfer_1d_2(pol, kx, epx_conv, epy_conv, epz_conv_i, type_complex=np.complex128, use_pinv=False):
    Kx = np.diag(kx)

    if pol == 0:
        A = Kx ** 2 - epy_conv
        eigenvalues, W = np.linalg.eig(A)
        eigenvalues += 0j  # to get positive square root
        q = eigenvalues ** 0.5
        Q = np.diag(q)
        V = W @ Q

    elif pol == 1:
        B = Kx @ epz_conv_i @ Kx - np.eye(epy_conv.shape[0], dtype=type_complex)

        eigenvalues, W = np.linalg.eig(epx_conv @ B)

        eigenvalues += 0j  # to get positive square root
        q = eigenvalues ** 0.5

        Q = np.diag(q)
        V = meeinv(epx_conv, use_pinv) @ W @ Q

    else:
        raise ValueError

    return W, V, q


def transfer_1d_3(k0, W, V, q, d, F, G, T, type_complex=np.complex128, use_pinv=False):
    ff_x = len(q)

    I = np.eye(ff_x, dtype=type_complex)

    X = np.diag(np.exp(-k0 * q * d))

    W_i = meeinv(W, use_pinv)
    V_i = meeinv(V, use_pinv)

    A = 0.5 * (W_i @ F + V_i @ G)
    B = 0.5 * (W_i @ F - V_i @ G)

    A_i = meeinv(A, use_pinv)

    F = W @ (I + X @ B @ A_i @ X)
    G = V @ (I - X @ B @ A_i @ X)
    T = T @ A_i @ X

    return X, F, G, T, A_i, B


def transfer_1d_4(pol, ff_x, F, G, T, kz_top, kz_bot, theta, n_top, n_bot, type_complex=np.complex128, use_pinv=False):
    Kz_top = np.diag(kz_top)
    kz_top = kz_top.reshape((1, ff_x))
    kz_bot = kz_bot.reshape((1, ff_x))

    delta_i0 = np.zeros(ff_x, dtype=type_complex)
    delta_i0[ff_x // 2] = 1

    if pol == 0:  # TE
        inc_term = 1j * n_top * np.cos(theta) * delta_i0
        T1 = meeinv(G + 1j * Kz_top @ F, use_pinv) @ (1j * Kz_top @ delta_i0 + inc_term)

    elif pol == 1:  # TM
        inc_term = 1j * delta_i0 * np.cos(theta) / n_top
        T1 = meeinv(G + 1j * Kz_top / (n_top ** 2) @ F, use_pinv) @ (1j * Kz_top / (n_top ** 2) @ delta_i0 + inc_term)
    else:
        raise ValueError

    R = (F @ T1 - delta_i0).reshape((1, ff_x))
    T = (T @ T1).reshape((1, ff_x))

    de_ri = (R * R.conj() * (kz_top / (n_top * np.cos(theta))).real).real

    if pol == 0:
        de_ti = (T * T.conj() * (kz_bot / (n_top * np.cos(theta))).real).real
        R_s = R
        R_p = np.zeros(R.shape)
        T_s = T
        T_p = np.zeros(T.shape)
        de_ri_s = de_ri
        de_ri_p = np.zeros(de_ri.shape)
        de_ti_s = de_ti
        de_ti_p = np.zeros(de_ri.shape)

    elif pol == 1:
        de_ti = (T * T.conj() * (kz_bot / n_bot ** 2 / (np.cos(theta) / n_top)).real).real
        R_s = np.zeros(R.shape)
        R_p = R
        T_s = np.zeros(T.shape)
        T_p = T
        de_ri_s = np.zeros(de_ri.shape)
        de_ri_p = de_ri
        de_ti_s = np.zeros(de_ri.shape)
        de_ti_p = de_ti
    else:
        raise ValueError

    res = {'R_s': R_s, 'R_p': R_p, 'T_s': T_s, 'T_p': T_p,
           'de_ri': de_ri, 'de_ri_s': de_ri_s, 'de_ri_p': de_ri_p,
           'de_ti': de_ti, 'de_ti_s': de_ti_s, 'de_ti_p': de_ti_p,
           }

    result = {'res': res}

    return result, T1


def transfer_1d_conical_1(kx, ky, n_top, n_bot, type_complex=np.complex128):
    ff_x = len(kx)
    ff_y = len(ky)
    ff_xy = ff_x * ff_y

    I = np.eye(ff_xy, dtype=type_complex)
    O = np.zeros((ff_xy, ff_xy), dtype=type_complex)

    kz_top = (n_top ** 2 - kx ** 2 - ky.reshape((-1, 1)) ** 2) ** 0.5
    kz_bot = (n_bot ** 2 - kx ** 2 - ky.reshape((-1, 1)) ** 2) ** 0.5

    kz_top = kz_top.flatten().conj()
    kz_bot = kz_bot.flatten().conj()

    varphi = np.arctan(ky.reshape((-1, 1)) / kx).flatten()
    Kz_bot = np.diag(kz_bot)

    big_F = np.block([[I, O], [O, 1j * Kz_bot / (n_bot ** 2)]])
    big_G = np.block([[1j * Kz_bot, O], [O, I]])
    big_T = np.eye(2 * ff_xy, dtype=type_complex)

    return kz_top, kz_bot, varphi, big_F, big_G, big_T


def transfer_1d_conical_2(kx, ky, epx_conv, epy_conv, epz_conv_i, type_complex=np.complex128, use_pinv=False):
    ff_x = len(kx)
    ff_y = len(ky)
    ff_xy = ff_x * ff_y

    I = np.eye(ff_xy, dtype=type_complex)

    Kx = np.diag(np.tile(kx, ff_y).flatten())
    Ky = np.diag(np.tile(ky.reshape((-1, 1)), ff_x).flatten())

    A = Kx ** 2 - epy_conv
    B = Kx @ epz_conv_i @ Kx - I

    Omega2_RL = Ky ** 2 + A
    Omega2_LR = Ky ** 2 + B @ epx_conv

    eigenvalues_1, W_1 = np.linalg.eig(Omega2_RL)
    eigenvalues_2, W_2 = np.linalg.eig(Omega2_LR)

    eigenvalues_1 += 0j  # to get positive square root
    eigenvalues_2 += 0j  # to get positive square root

    q_1 = eigenvalues_1 ** 0.5
    q_2 = eigenvalues_2 ** 0.5

    Q_1 = np.diag(q_1)
    Q_2 = np.diag(q_2)

    A_i = meeinv(A, use_pinv)
    B_i = meeinv(B, use_pinv)

    V_11 = A_i @ W_1 @ Q_1
    V_12 = Ky @ A_i @ Kx @ W_2
    V_21 = Ky @ B_i @ Kx @ epz_conv_i @ W_1
    V_22 = B_i @ W_2 @ Q_2

    W = np.block([W_1, W_2])
    V = np.block([[V_11, V_12],
                  [V_21, V_22]])
    q = np.hstack([q_1, q_2])

    return W, V, q


def transfer_1d_conical_3(k0, W, V, q, d, varphi, big_F, big_G, big_T, type_complex=np.complex128, use_pinv=False):
    ff_xy = len(q) // 2
    I = np.eye(ff_xy, dtype=type_complex)
    O = np.zeros((ff_xy, ff_xy), dtype=type_complex)

    q_1 = q[:ff_xy]
    q_2 = q[ff_xy:]

    W_1 = W[:, :ff_xy]
    W_2 = W[:, ff_xy:]

    V_11 = V[:ff_xy, :ff_xy]
    V_12 = V[:ff_xy, ff_xy:]
    V_21 = V[ff_xy:, :ff_xy]
    V_22 = V[ff_xy:, ff_xy:]

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

    big_I = np.eye(2 * (len(I)), dtype=type_complex)
    big_X = np.block([[X_1, O], [O, X_2]])
    big_W = np.block([[V_ss, V_sp], [W_ps, W_pp]])
    big_V = np.block([[W_ss, W_sp], [V_ps, V_pp]])

    big_W_i = meeinv(big_W, use_pinv)
    big_V_i = meeinv(big_V, use_pinv)

    big_A = 0.5 * (big_W_i @ big_F + big_V_i @ big_G)
    big_B = 0.5 * (big_W_i @ big_F - big_V_i @ big_G)

    big_A_i = meeinv(big_A, use_pinv)

    big_F = big_W @ (big_I + big_X @ big_B @ big_A_i @ big_X)
    big_G = big_V @ (big_I - big_X @ big_B @ big_A_i @ big_X)

    big_T = big_T @ big_A_i @ big_X

    return big_X, big_F, big_G, big_T, big_A_i, big_B


def transfer_1d_conical_4(ff_x, ff_y, big_F, big_G, big_T, kz_top, kz_bot, psi, theta, n_top, n_bot,
                          type_complex=np.complex128, use_pinv=False):

    ff_xy = ff_x * ff_y

    Kz_top = np.diag(kz_top)
    kz_top = kz_top.reshape((ff_y, ff_x))
    kz_bot = kz_bot.reshape((ff_y, ff_x))

    I = np.eye(ff_xy, dtype=type_complex)
    O = np.zeros((ff_xy, ff_xy), dtype=type_complex)

    big_F_11 = big_F[:ff_xy, :ff_xy]
    big_F_12 = big_F[:ff_xy, ff_xy:]
    big_F_21 = big_F[ff_xy:, :ff_xy]
    big_F_22 = big_F[ff_xy:, ff_xy:]

    big_G_11 = big_G[:ff_xy, :ff_xy]
    big_G_12 = big_G[:ff_xy, ff_xy:]
    big_G_21 = big_G[ff_xy:, :ff_xy]
    big_G_22 = big_G[ff_xy:, ff_xy:]

    delta_i0 = np.zeros((ff_xy, 1), dtype=type_complex)
    delta_i0[ff_xy // 2, 0] = 1

    # Final Equation in form of AX=B
    final_A = np.block(
        [
            [I, O, -big_F_11, -big_F_12],
            [O, -1j * Kz_top / (n_top ** 2), -big_F_21, -big_F_22],
            [-1j * Kz_top, O, -big_G_11, -big_G_12],
            [O, I, -big_G_21, -big_G_22],
        ]
    )
    final_B = np.block(
        [
            [-np.sin(psi) * delta_i0],
            [np.cos(psi) * np.cos(theta) * delta_i0],
            [-1j * np.sin(psi) * n_top * np.cos(theta) * delta_i0],
            [-1j * n_top * np.cos(psi) * delta_i0]
        ]
    )

    final_A_inv = meeinv(final_A, use_pinv)
    final_RT = final_A_inv @ final_B

    R_s = final_RT[:ff_xy, :].reshape((ff_y, ff_x))
    R_p = final_RT[ff_xy: 2 * ff_xy, :].reshape((ff_y, ff_x))

    big_T1 = final_RT[2 * ff_xy:, :]
    big_T_tetm = big_T.copy()
    big_T = big_T @ big_T1

    T_s = big_T[:ff_xy, :].reshape((ff_y, ff_x))
    T_p = big_T[ff_xy:, :].reshape((ff_y, ff_x))

    de_ri_s = (R_s * R_s.conj() * (kz_top / (n_top * np.cos(theta))).real).real
    de_ri_p = (R_p * R_p.conj() * (kz_top / n_top ** 2 / (n_top * np.cos(theta))).real).real

    de_ti_s = (T_s * T_s.conj() * (kz_bot / (n_top * np.cos(theta))).real).real
    de_ti_p = (T_p * T_p.conj() * (kz_bot / n_bot ** 2 / (n_top * np.cos(theta))).real).real

    de_ri = de_ri_s + de_ri_p
    de_ti = de_ti_s + de_ti_p

    res = {'R_s': R_s, 'R_p': R_p, 'T_s': T_s, 'T_p': T_p,
           'de_ri_s': de_ri_s, 'de_ri_p': de_ri_p, 'de_ri': de_ri,
           'de_ti_s': de_ti_s, 'de_ti_p': de_ti_p, 'de_ti': de_ti}

    # TE TM incidence
    psi_tm = np.array(0, dtype=type_complex)
    final_B_tm = np.block(
        [
            [-np.sin(psi_tm) * delta_i0],
            [np.cos(psi_tm) * np.cos(theta) * delta_i0],
            [-1j * np.sin(psi_tm) * n_top * np.cos(theta) * delta_i0],
            [-1j * n_top * np.cos(psi_tm) * delta_i0]
        ]
    )

    psi_te = np.array(np.pi / 2, dtype=type_complex)
    final_B_te = np.block(
        [
            [-np.sin(psi_te) * delta_i0],
            [np.cos(psi_te) * np.cos(theta) * delta_i0],
            [-1j * np.sin(psi_te) * n_top * np.cos(theta) * delta_i0],
            [-1j * n_top * np.cos(psi_te) * delta_i0]
        ]
    )

    final_B_tetm = np.hstack([final_B_te, final_B_tm])
    final_RT_tetm = final_A_inv @ final_B_tetm

    R_s_tetm = final_RT_tetm[:ff_xy, :].T.reshape((2, ff_y, ff_x))
    R_p_tetm = final_RT_tetm[ff_xy: 2 * ff_xy, :].T.reshape((2, ff_y, ff_x))

    big_T1_tetm = final_RT_tetm[2 * ff_xy:, :]
    big_T_tetm = big_T_tetm @ big_T1_tetm

    T_s_tetm = big_T_tetm[:ff_xy, :].T.reshape((2, ff_y, ff_x))
    T_p_tetm = big_T_tetm[ff_xy:, :].T.reshape((2, ff_y, ff_x))

    de_ri_s_tetm = (R_s_tetm * R_s_tetm.conj() * (kz_top / (n_top * np.cos(theta))).real).real
    de_ri_p_tetm = (R_p_tetm * R_p_tetm.conj() * (kz_top / n_top ** 2 / (n_top * np.cos(theta))).real).real

    de_ti_s_tetm = (T_s_tetm * T_s_tetm.conj() * (kz_bot / (n_top * np.cos(theta))).real).real
    de_ti_p_tetm = (T_p_tetm * T_p_tetm.conj() * (kz_bot / n_bot ** 2 / (n_top * np.cos(theta))).real).real

    de_ri_tetm = de_ri_s_tetm + de_ri_p_tetm
    de_ti_tetm = de_ti_s_tetm + de_ti_p_tetm

    res_te_inc = {'R_s': R_s_tetm[0], 'R_p': R_p_tetm[0], 'T_s': T_s_tetm[0], 'T_p': T_p_tetm[0],
                  'de_ri_s': de_ri_s_tetm[0], 'de_ri_p': de_ri_p_tetm[0], 'de_ri': de_ri_tetm[0],
                  'de_ti_s': de_ti_s_tetm[0], 'de_ti_p': de_ti_p_tetm[0], 'de_ti': de_ti_tetm[0]}

    res_tm_inc = {'R_s': R_s_tetm[1], 'R_p': R_p_tetm[1], 'T_s': T_s_tetm[1], 'T_p': T_p_tetm[1],
                  'de_ri_s': de_ri_s_tetm[1], 'de_ri_p': de_ri_p_tetm[1], 'de_ri': de_ri_tetm[1],
                  'de_ti_s': de_ti_s_tetm[1], 'de_ti_p': de_ti_p_tetm[1], 'de_ti': de_ti_tetm[1]}

    result = {'res': res, 'res_tm_inc': res_tm_inc, 'res_te_inc': res_te_inc}
    big_T1_all = np.stack((big_T1, big_T1_tetm[:, 0:1], big_T1_tetm[:, 1:2]))

    return result, big_T1_all


def transfer_2d_1(kx, ky, n_top, n_bot, type_complex=np.complex128):
    ff_x = len(kx)
    ff_y = len(ky)
    ff_xy = ff_x * ff_y

    I = np.eye(ff_xy, dtype=type_complex)
    O = np.zeros((ff_xy, ff_xy), dtype=type_complex)

    kz_top = (n_top ** 2 - kx ** 2 - ky.reshape((-1, 1)) ** 2) ** 0.5
    kz_bot = (n_bot ** 2 - kx ** 2 - ky.reshape((-1, 1)) ** 2) ** 0.5

    kz_top = kz_top.flatten().conj()
    kz_bot = kz_bot.flatten().conj()

    varphi = np.arctan(ky.reshape((-1, 1)) / kx).flatten()
    Kz_bot = np.diag(kz_bot)

    big_F = np.block([[I, O], [O, 1j * Kz_bot / (n_bot ** 2)]])
    big_G = np.block([[1j * Kz_bot, O], [O, I]])
    big_T = np.eye(2 * ff_xy, dtype=type_complex)

    return kz_top, kz_bot, varphi, big_F, big_G, big_T


def transfer_2d_2(kx, ky, epx_conv, epy_conv, epz_conv_i, type_complex=np.complex128, use_pinv=False):
    ff_x = len(kx)
    ff_y = len(ky)
    ff_xy = ff_x * ff_y

    I = np.eye(ff_xy, dtype=type_complex)

    Kx = np.diag(np.tile(kx, ff_y).flatten())
    Ky = np.diag(np.tile(ky.reshape((-1, 1)), ff_x).flatten())

    B = Kx @ epz_conv_i @ Kx - I
    D = Ky @ epz_conv_i @ Ky - I

    Omega2_LR = np.block(
        [
            [Ky ** 2 + B @ epx_conv, Kx @ (epz_conv_i @ Ky @ epy_conv - Ky)],
            [Ky @ (epz_conv_i @ Kx @ epx_conv - Kx), Kx ** 2 + D @ epy_conv]
        ])

    eigenvalues, W = np.linalg.eig(Omega2_LR)

    eigenvalues += 0j  # to get positive square root
    q = eigenvalues ** 0.5

    Q = np.diag(q)
    Q_i = meeinv(Q, use_pinv)

    Omega_R = np.block(
        [
            [-Kx @ Ky, Kx ** 2 - epy_conv],
            [epx_conv - Ky ** 2, Ky @ Kx]
        ]
    )

    V = Omega_R @ W @ Q_i

    return W, V, q


def transfer_2d_3(k0, W, V, q, d, varphi, big_F, big_G, big_T, type_complex=np.complex128, use_pinv=False):
    ff_xy = len(q) // 2

    I = np.eye(ff_xy, dtype=type_complex)
    O = np.zeros((ff_xy, ff_xy), dtype=type_complex)

    q_1 = q[:ff_xy]
    q_2 = q[ff_xy:]

    W_11 = W[:ff_xy, :ff_xy]
    W_12 = W[:ff_xy, ff_xy:]
    W_21 = W[ff_xy:, :ff_xy]
    W_22 = W[ff_xy:, ff_xy:]

    V_11 = V[:ff_xy, :ff_xy]
    V_12 = V[:ff_xy, ff_xy:]
    V_21 = V[ff_xy:, :ff_xy]
    V_22 = V[ff_xy:, ff_xy:]

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

    big_I = np.eye(2 * (len(I)), dtype=type_complex)
    big_X = np.block([[X_1, O], [O, X_2]])
    big_W = np.block([[W_ss, W_sp], [W_ps, W_pp]])
    big_V = np.block([[V_ss, V_sp], [V_ps, V_pp]])

    big_W_i = meeinv(big_W, use_pinv)
    big_V_i = meeinv(big_V, use_pinv)

    big_A = 0.5 * (big_W_i @ big_F + big_V_i @ big_G)
    big_B = 0.5 * (big_W_i @ big_F - big_V_i @ big_G)

    big_A_i = meeinv(big_A, use_pinv)

    big_F = big_W @ (big_I + big_X @ big_B @ big_A_i @ big_X)
    big_G = big_V @ (big_I - big_X @ big_B @ big_A_i @ big_X)

    big_T = big_T @ big_A_i @ big_X

    return big_X, big_F, big_G, big_T, big_A_i, big_B


def transfer_2d_4(ff_x, ff_y, big_F, big_G, big_T, kz_top, kz_bot, psi, theta, n_top, n_bot,
                  type_complex=np.complex128, use_pinv=False):

    ff_xy = ff_x * ff_y

    Kz_top = np.diag(kz_top)
    kz_top = kz_top.reshape((ff_y, ff_x))
    kz_bot = kz_bot.reshape((ff_y, ff_x))

    I = np.eye(ff_xy, dtype=type_complex)
    O = np.zeros((ff_xy, ff_xy), dtype=type_complex)

    big_F_11 = big_F[:ff_xy, :ff_xy]
    big_F_12 = big_F[:ff_xy, ff_xy:]
    big_F_21 = big_F[ff_xy:, :ff_xy]
    big_F_22 = big_F[ff_xy:, ff_xy:]

    big_G_11 = big_G[:ff_xy, :ff_xy]
    big_G_12 = big_G[:ff_xy, ff_xy:]
    big_G_21 = big_G[ff_xy:, :ff_xy]
    big_G_22 = big_G[ff_xy:, ff_xy:]

    delta_i0 = np.zeros((ff_xy, 1), dtype=type_complex)
    delta_i0[ff_xy // 2, 0] = 1

    # Final Equation in form of AX=B
    final_A = np.block(
        [
            [I, O, -big_F_11, -big_F_12],
            [O, -1j * Kz_top / (n_top ** 2), -big_F_21, -big_F_22],
            [-1j * Kz_top, O, -big_G_11, -big_G_12],
            [O, I, -big_G_21, -big_G_22],
        ]
    )

    final_B = np.block(
        [
            [-np.sin(psi) * delta_i0],
            [np.cos(psi) * np.cos(theta) * delta_i0],
            [-1j * np.sin(psi) * n_top * np.cos(theta) * delta_i0],
            [-1j * n_top * np.cos(psi) * delta_i0]
        ]
    )

    final_A_inv = meeinv(final_A, use_pinv)
    final_RT = final_A_inv @ final_B

    R_s = final_RT[:ff_xy, :].reshape((ff_y, ff_x))
    R_p = final_RT[ff_xy: 2 * ff_xy, :].reshape((ff_y, ff_x))

    big_T1 = final_RT[2 * ff_xy:, :]
    big_T_tetm = big_T.copy()
    big_T = big_T @ big_T1

    T_s = big_T[:ff_xy, :].reshape((ff_y, ff_x))
    T_p = big_T[ff_xy:, :].reshape((ff_y, ff_x))

    de_ri_s = (R_s * R_s.conj() * (kz_top / (n_top * np.cos(theta))).real).real
    de_ri_p = (R_p * R_p.conj() * (kz_top / n_top ** 2 / (n_top * np.cos(theta))).real).real

    de_ti_s = (T_s * T_s.conj() * (kz_bot / (n_top * np.cos(theta))).real).real
    de_ti_p = (T_p * T_p.conj() * (kz_bot / n_bot ** 2 / (n_top * np.cos(theta))).real).real

    de_ri = de_ri_s + de_ri_p
    de_ti = de_ti_s + de_ti_p

    res = {'R_s': R_s, 'R_p': R_p, 'T_s': T_s, 'T_p': T_p,
           'de_ri_s': de_ri_s, 'de_ri_p': de_ri_p, 'de_ri': de_ri,
           'de_ti_s': de_ti_s, 'de_ti_p': de_ti_p, 'de_ti': de_ti}

    # TE TM incidence
    psi_tm = np.array(0, dtype=type_complex)
    final_B_tm = np.block(
        [
            [-np.sin(psi_tm) * delta_i0],
            [np.cos(psi_tm) * np.cos(theta) * delta_i0],
            [-1j * np.sin(psi_tm) * n_top * np.cos(theta) * delta_i0],
            [-1j * n_top * np.cos(psi_tm) * delta_i0]
        ]
    )

    psi_te = np.array(np.pi/2, dtype=type_complex)
    final_B_te = np.block(
        [
            [-np.sin(psi_te) * delta_i0],
            [np.cos(psi_te) * np.cos(theta) * delta_i0],
            [-1j * np.sin(psi_te) * n_top * np.cos(theta) * delta_i0],
            [-1j * n_top * np.cos(psi_te) * delta_i0]
        ]
    )

    final_B_tetm = np.hstack([final_B_te, final_B_tm])
    final_RT_tetm = final_A_inv @ final_B_tetm

    R_s_tetm = final_RT_tetm[:ff_xy, :].T.reshape((2, ff_y, ff_x))
    R_p_tetm = final_RT_tetm[ff_xy: 2 * ff_xy, :].T.reshape((2, ff_y, ff_x))

    big_T1_tetm = final_RT_tetm[2 * ff_xy:, :]
    big_T_tetm = big_T_tetm @ big_T1_tetm

    T_s_tetm = big_T_tetm[:ff_xy, :].T.reshape((2, ff_y, ff_x))
    T_p_tetm = big_T_tetm[ff_xy:, :].T.reshape((2, ff_y, ff_x))

    de_ri_s_tetm = (R_s_tetm * R_s_tetm.conj() * (kz_top / (n_top * np.cos(theta))).real).real
    de_ri_p_tetm = (R_p_tetm * R_p_tetm.conj() * (kz_top / n_top ** 2 / (n_top * np.cos(theta))).real).real

    de_ti_s_tetm = (T_s_tetm * T_s_tetm.conj() * (kz_bot / (n_top * np.cos(theta))).real).real
    de_ti_p_tetm = (T_p_tetm * T_p_tetm.conj() * (kz_bot / n_bot ** 2 / (n_top * np.cos(theta))).real).real

    de_ri_tetm = de_ri_s_tetm + de_ri_p_tetm
    de_ti_tetm = de_ti_s_tetm + de_ti_p_tetm

    res_te_inc = {'R_s': R_s_tetm[0], 'R_p': R_p_tetm[0], 'T_s': T_s_tetm[0], 'T_p': T_p_tetm[0],
                  'de_ri_s': de_ri_s_tetm[0], 'de_ri_p': de_ri_p_tetm[0], 'de_ri': de_ri_tetm[0],
                  'de_ti_s': de_ti_s_tetm[0], 'de_ti_p': de_ti_p_tetm[0], 'de_ti': de_ti_tetm[0]}

    res_tm_inc = {'R_s': R_s_tetm[1], 'R_p': R_p_tetm[1], 'T_s': T_s_tetm[1], 'T_p': T_p_tetm[1],
                  'de_ri_s': de_ri_s_tetm[1], 'de_ri_p': de_ri_p_tetm[1], 'de_ri': de_ri_tetm[1],
                  'de_ti_s': de_ti_s_tetm[1], 'de_ti_p': de_ti_p_tetm[1], 'de_ti': de_ti_tetm[1]}

    result = {'res': res, 'res_tm_inc': res_tm_inc, 'res_te_inc': res_te_inc}
    big_T1_all = np.stack((big_T1, big_T1_tetm[:, 0:1], big_T1_tetm[:, 1:2]))

    return result, big_T1_all