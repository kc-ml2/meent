import numpy as np


def transfer_1d_1(ff, polarization, k0, n_I, n_II, kx_vector, theta, delta_i0, fourier_order,
                  type_complex=np.complex128):

    # kx_vector = k0 * (n_I * np.sin(theta) + fourier_indices * (wavelength / period[0])).astype(type_complex)

    kz_top = (k0 ** 2 * n_I ** 2 - kx_vector ** 2) ** 0.5
    kz_bot = (k0 ** 2 * n_II ** 2 - kx_vector ** 2) ** 0.5

    kz_top = kz_top.conjugate()
    kz_bot = kz_bot.conjugate()

    Kx = np.diag(kx_vector / k0)

    f = np.eye(ff, dtype=type_complex)

    if polarization == 0:  # TE
        Y_I = np.diag(kz_top / k0)
        Y_II = np.diag(kz_bot / k0)

        YZ_I = Y_I
        g = 1j * Y_II
        inc_term = 1j * n_I * np.cos(theta) * delta_i0

    elif polarization == 1:  # TM
        Z_I = np.diag(kz_top / (k0 * n_I ** 2))
        Z_II = np.diag(kz_bot / (k0 * n_II ** 2))

        YZ_I = Z_I
        g = 1j * Z_II
        inc_term = 1j * delta_i0 * np.cos(theta) / n_I  # tODO: inc term?

    else:
        raise ValueError

    T = np.eye(2 * fourier_order[0] + 1, dtype=type_complex)

    return kx_vector, Kx, kz_top, kz_bot, f, YZ_I, g, inc_term, T


def transfer_1d_2(pol, Kx, epx_conv, epy_conv, epz_conv_i, type_complex=np.complex128):

    if pol == 0:
        A = Kx ** 2 - epy_conv
        eigenvalues, W = np.linalg.eig(A)
        eigenvalues += 0j  # to get positive square root
        q = eigenvalues ** 0.5
        Q = np.diag(q)
        V = W @ Q

    elif pol == 1:
        B = Kx @ epz_conv_i @ Kx - np.eye(epy_conv.shape[0], dtype=type_complex)

        # eigenvalues, W = np.linalg.eig(E_conv @ B)
        eigenvalues, W = np.linalg.eig(epx_conv @ B)

        eigenvalues += 0j  # to get positive square root
        q = eigenvalues ** 0.5

        Q = np.diag(q)
        V = np.linalg.inv(epx_conv) @ W @ Q

    else:
        raise ValueError

    return W, V, q


def transfer_1d_3(k0, W, V, q, d, f, g, T, type_complex=np.complex128):

    ff_x = len(q)

    I = np.eye(ff_x, dtype=type_complex)

    X = np.diag(np.exp(-k0 * q * d))

    W_i = np.linalg.inv(W)
    V_i = np.linalg.inv(V)

    a = 0.5 * (W_i @ f + V_i @ g)
    b = 0.5 * (W_i @ f - V_i @ g)

    a_i = np.linalg.inv(a)

    f = W @ (I+ X @ b @ a_i @ X)
    g = V @ (I- X @ b @ a_i @ X)
    T = T @ a_i @ X

    return X, f, g, T, a_i, b


def transfer_1d_4(k0, f, g, T, YZ_I, kz_top, delta_i0, inc_term,  n_I, n_II, theta, pol, kz_bot):

    T1 = np.linalg.inv(g + 1j * YZ_I @ f) @ (1j * YZ_I @ delta_i0 + inc_term)
    R = f @ T1 - delta_i0
    T = T @ T1

    de_ri = np.real(R * np.conj(R) * kz_top / (k0 * n_I * np.cos(theta)))
    if pol == 0:
        de_ti = T * np.conj(T) * np.real(kz_bot / (k0 * n_I * np.cos(theta)))
    elif pol == 1:
        de_ti = T * np.conj(T) * np.real(kz_bot / n_II ** 2) / (k0 * np.cos(theta) / n_I)
    else:
        raise ValueError

    return de_ri.real, de_ti.real, T1


def transfer_1d_conical_1(k0, ff_x, n_I, n_II, kx_vector, ky_vector, type_complex=np.complex128):

    I = np.eye(ff_x, dtype=type_complex)
    O = np.zeros((ff_x, ff_x), dtype=type_complex)

    # kx_vector = k0 * (n_I * np.sin(theta) * np.cos(phi) + fourier_indices * (wavelength / period[0])
    #                   ).astype(type_complex)

    # ky = k0 * n_I * np.sin(theta) * np.sin(phi)  # TODO: check ky is equal to ky_vector

    kz_top = (k0 ** 2 * n_I ** 2 - kx_vector ** 2 - ky_vector ** 2) ** 0.5
    kz_bot = (k0 ** 2 * n_II ** 2 - kx_vector ** 2 - ky_vector ** 2) ** 0.5

    kz_top = kz_top.conjugate()
    kz_bot = kz_bot.conjugate()

    Kx = np.diag(kx_vector / k0)
    Ky = np.diag(ky_vector / k0)

    varphi = np.arctan(ky_vector / kx_vector)

    Y_I = np.diag(kz_top / k0)
    Y_II = np.diag(kz_bot / k0)

    Z_I = np.diag(kz_top / (k0 * n_I ** 2))
    Z_II = np.diag(kz_bot / (k0 * n_II ** 2))

    big_F = np.block([[I, O], [O, 1j * Z_II]])
    big_G = np.block([[1j * Y_II, O], [O, I]])

    big_T = np.eye(2 * ff_x, dtype=type_complex)

    return Kx, Ky, kz_top, kz_bot, varphi, Y_I, Y_II, Z_I, Z_II, big_F, big_G, big_T


def transfer_1d_conical_2(Kx, Ky, epx_conv, epy_conv, epz_conv_i,
                          type_complex=np.complex128):

    I = np.eye(len(Kx), dtype=type_complex)

    A = Kx ** 2 - epy_conv
    B = Kx @ epz_conv_i @ Kx - I

    A_i = np.linalg.inv(A)
    B_i = np.linalg.inv(B)

    # Todo: remove k0
    # to_decompose_W_1 = (ky/k0) ** 2 * I + A
    to_decompose_W_1 = Ky ** 2 * I + A

    # to_decompose_W_2 = (ky/k0) ** 2 * I + B @ epx_conv
    to_decompose_W_2 = Ky ** 2 * I + B @ epx_conv

    eigenvalues_1, W_1 = np.linalg.eig(to_decompose_W_1)
    eigenvalues_2, W_2 = np.linalg.eig(to_decompose_W_2)
    eigenvalues_1 += 0j  # to get positive square root
    eigenvalues_2 += 0j  # to get positive square root

    q_1 = eigenvalues_1 ** 0.5
    q_2 = eigenvalues_2 ** 0.5

    Q_1 = np.diag(q_1)
    Q_2 = np.diag(q_2)

    # V_11 = A_i @ W_1 @ Q_1
    # V_12 = (ky / k0) * A_i @ Kx @ W_2
    # V_21 = (ky / k0) * B_i @ Kx @ epz_conv_i @ W_1
    # V_22 = B_i @ W_2 @ Q_2

    V_11 = A_i @ W_1 @ Q_1
    V_12 = Ky * A_i @ Kx @ W_2
    V_21 = Ky * B_i @ Kx @ epz_conv_i @ W_1
    V_22 = B_i @ W_2 @ Q_2

    W = np.block([W_1, W_2])
    V = np.block([[V_11, V_12],
                  [V_21, V_22]])
    q = np.hstack([q_1, q_2])

    # return W_1, W_2, V_11, V_12, V_21, V_22, q_1, q_2

    return W, V, q

    #
    # X_1 = np.diag(np.exp(-k0 * q_1 * d))
    # X_2 = np.diag(np.exp(-k0 * q_2 * d))
    #
    # F_c = np.diag(np.cos(varphi))
    # F_s = np.diag(np.sin(varphi))
    #
    # V_ss = F_c @ V_11
    # V_sp = F_c @ V_12 - F_s @ W_2
    # W_ss = F_c @ W_1 + F_s @ V_21
    # W_sp = F_s @ V_22
    # W_ps = F_s @ V_11
    # W_pp = F_c @ W_2 + F_s @ V_12
    # V_ps = F_c @ V_21 - F_s @ W_1
    # V_pp = F_c @ V_22
    #
    # big_I = np.eye(2 * (len(I)), dtype=type_complex)
    # big_X = np.block([[X_1, O], [O, X_2]])
    # big_W = np.block([[V_ss, V_sp], [W_ps, W_pp]])
    # big_V = np.block([[W_ss, W_sp], [V_ps, V_pp]])
    #
    # big_W_i = np.linalg.inv(big_W)
    # big_V_i = np.linalg.inv(big_V)
    #
    # big_A = 0.5 * (big_W_i @ big_F + big_V_i @ big_G)
    # big_B = 0.5 * (big_W_i @ big_F - big_V_i @ big_G)
    #
    # big_A_i = np.linalg.inv(big_A)
    #
    # big_F = big_W @ (big_I + big_X @ big_B @ big_A_i @ big_X)
    # big_G = big_V @ (big_I - big_X @ big_B @ big_A_i @ big_X)
    #
    # big_T = big_T @ big_A_i @ big_X
    #
    # return big_X, big_F, big_G, big_T, big_A_i, big_B, W_1, W_2, V_11, V_12, V_21, V_22, q_1, q_2


def transfer_1d_conical_3(k0, W, V, q, d, varphi, big_F, big_G, big_T,
                          type_complex=np.complex128):

    ff_x = len(W)

    I = np.eye(ff_x, dtype=type_complex)
    O = np.zeros((ff_x, ff_x), dtype=type_complex)

    W_1 = W[:ff_x]
    W_2 = W[ff_x:]

    V_11 = V[:ff_x, :ff_x]
    V_12 = V[:ff_x, ff_x:]
    V_21 = V[ff_x:, :ff_x]
    V_22 = V[ff_x:, ff_x:]

    q_1 = q[:ff_x]
    q_2 = q[ff_x:]

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

    big_W_i = np.linalg.inv(big_W)
    big_V_i = np.linalg.inv(big_V)

    big_A = 0.5 * (big_W_i @ big_F + big_V_i @ big_G)
    big_B = 0.5 * (big_W_i @ big_F - big_V_i @ big_G)

    big_A_i = np.linalg.inv(big_A)

    big_F = big_W @ (big_I + big_X @ big_B @ big_A_i @ big_X)
    big_G = big_V @ (big_I - big_X @ big_B @ big_A_i @ big_X)

    big_T = big_T @ big_A_i @ big_X

    return big_X, big_F, big_G, big_T, big_A_i, big_B


def transfer_1d_conical_4(k0, big_F, big_G, big_T, Z_I, Y_I, kz_top, kz_bot, psi, theta,  n_I, n_II,
                          type_complex=np.complex128):

    ff_xy = len(big_F) // 2

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

    delta_i0 = np.zeros(ff_xy, dtype=type_complex)
    delta_i0[ff_xy // 2] = 1

    # Final Equation in form of AX=B
    final_A = np.block(
        [
            [I, O, -big_F_11, -big_F_12],
            [O, -1j * Z_I, -big_F_21, -big_F_22],
            [-1j * Y_I, O, -big_G_11, -big_G_12],
            [O, I, -big_G_21, -big_G_22],
        ]
    )

    final_B = np.hstack([
        [-np.sin(psi) * delta_i0],
        [-np.cos(psi) * np.cos(theta) * delta_i0],
        [-1j * np.sin(psi) * n_I * np.cos(theta) * delta_i0],
        [1j * n_I * np.cos(psi) * delta_i0]
    ]).T

    final_RT = np.linalg.inv(final_A) @ final_B

    R_s = final_RT[:ff_xy, :].flatten()
    R_p = final_RT[ff_xy:2 * ff_xy, :].flatten()

    big_T1 = final_RT[2 * ff_xy:, :]
    big_T = big_T @ big_T1

    T_s = big_T[:ff_xy, :].flatten()
    T_p = big_T[ff_xy:, :].flatten()

    de_ri = R_s * np.conj(R_s) * np.real(kz_top / (k0 * n_I * np.cos(theta))) \
            + R_p * np.conj(R_p) * np.real((kz_top / n_I ** 2) / (k0 * n_I * np.cos(theta)))

    de_ti = T_s * np.conj(T_s) * np.real(kz_bot / (k0 * n_I * np.cos(theta))) \
            + T_p * np.conj(T_p) * np.real((kz_bot / n_II ** 2) / (k0 * n_I * np.cos(theta)))

    return de_ri.real, de_ti.real, big_T1


def transfer_2d_1(k0, ff_x, ff_y, kx_vector, ky_vector, n_I, n_II, type_complex=np.complex128):

    ff_xy = ff_x * ff_y

    I = np.eye(ff_xy, dtype=type_complex)
    O = np.zeros((ff_xy, ff_xy), dtype=type_complex)

    kz_top = (k0 ** 2 * n_I ** 2 - kx_vector ** 2 - ky_vector.reshape((-1, 1)) ** 2) ** 0.5
    kz_bot = (k0 ** 2 * n_II ** 2 - kx_vector ** 2 - ky_vector.reshape((-1, 1)) ** 2) ** 0.5

    kz_top = kz_top.flatten().conjugate()
    kz_bot = kz_bot.flatten().conjugate()

    varphi = np.arctan(ky_vector.reshape((-1, 1)) / kx_vector).flatten()

    Kz_bot_s = np.diag(kz_bot / k0)
    Kz_bot_p = np.diag(kz_bot / (k0 * n_II ** 2))

    big_F = np.block([[I, O], [O, 1j * Kz_bot_p]])
    big_G = np.block([[1j * Kz_bot_s, O], [O, I]])

    Kz_bot = np.diag(kz_bot / k0)
    Kz_bot_p = np.diag(kz_bot / (k0 * n_II ** 2))

    big_F = np.block([[I, O], [O, 1j * Kz_bot_p]])
    big_G = np.block([[1j * Kz_bot_s, O], [O, I]])



    big_T = np.eye(2 * ff_xy, dtype=type_complex)

    # return Kx, Ky, kz_top, kz_bot, varphi, Y_I, Y_II, Z_I, Z_II, big_F, big_G, big_T
    return kz_top, kz_bot, varphi, big_F, big_G, big_T


def transfer_2d_2(k0, kx, ky, epx_conv, epy_conv, epz_conv_i, type_complex=np.complex128):

    ff_x = len(kx)
    ff_y = len(ky)

    I = np.eye(ff_y * ff_x, dtype=type_complex)

    Kx = np.diag(np.tile(kx, ff_y).flatten()) / k0
    Ky = np.diag(np.tile(ky.reshape((-1, 1)), ff_x).flatten()) / k0

    B = Kx @ epz_conv_i @ Kx - I
    D = Ky @ epz_conv_i @ Ky - I


    S2_from_S = np.block(
        [
            [Ky ** 2 + B @ epx_conv, Kx @ (epz_conv_i @ Ky @ epy_conv - Ky)],
            [Ky @ (epz_conv_i @ Kx @ epx_conv - Kx), Kx ** 2 + D @ epy_conv]
        ])

    eigenvalues, W = np.linalg.eig(S2_from_S)
    eigenvalues += 0j  # to get positive square root
    q = eigenvalues ** 0.5

    Q = np.diag(q)
    Q_i = np.linalg.inv(Q)

    U1_from_S = np.block(
        [
            [-Kx @ Ky, Kx ** 2 - epy_conv],
            [epx_conv - Ky ** 2, Ky @ Kx]
        ]
    )

    V = U1_from_S @ W @ Q_i

    return W, V, q


def transfer_2d_3(k0, W, V, q, d, varphi, big_F, big_G, big_T, type_complex=np.complex128):

    ff_xy = len(q)//2

    I = np.eye(ff_xy, dtype=type_complex)
    O = np.zeros((ff_xy, ff_xy), dtype=type_complex)

    q1 = q[:ff_xy]
    q2 = q[ff_xy:]

    W_11 = W[:ff_xy, :ff_xy]
    W_12 = W[:ff_xy, ff_xy:]
    W_21 = W[ff_xy:, :ff_xy]
    W_22 = W[ff_xy:, ff_xy:]

    V_11 = V[:ff_xy, :ff_xy]
    V_12 = V[:ff_xy, ff_xy:]
    V_21 = V[ff_xy:, :ff_xy]
    V_22 = V[ff_xy:, ff_xy:]

    X_1 = np.diag(np.exp(-k0 * q1 * d))
    X_2 = np.diag(np.exp(-k0 * q2 * d))

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

    big_W_i = np.linalg.inv(big_W)
    big_V_i = np.linalg.inv(big_V)

    big_A = 0.5 * (big_W_i @ big_F + big_V_i @ big_G)
    big_B = 0.5 * (big_W_i @ big_F - big_V_i @ big_G)

    big_A_i = np.linalg.inv(big_A)

    big_F = big_W @ (big_I + big_X @ big_B @ big_A_i @ big_X)
    big_G = big_V @ (big_I - big_X @ big_B @ big_A_i @ big_X)

    big_T = big_T @ big_A_i @ big_X

    return big_X, big_F, big_G, big_T, big_A_i, big_B


def transfer_2d_4(k0, big_F, big_G, big_T, kz_top, kz_bot, psi, theta, n_I, n_II,
                  type_complex=np.complex128):

    ff_xy = len(big_F) // 2

    Kz_top_s = np.diag(kz_top / k0)  # Y_I
    Kz_top_p = np.diag(kz_top / (k0 * n_I ** 2))  # Z_I

    Kz_bot_s = np.diag(kz_bot / k0)  # Y_II
    Kz_bot_p = np.diag(kz_bot / (k0 * n_II ** 2))  # Z_II

    kz_bot_s = np.diag(kz_bot / k0)  # Y_II
    kz_bot_p = np.diag(kz_bot / (k0 * n_II ** 2))  # Z_II

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
            [O, -1j * Kz_top_p, -big_F_21, -big_F_22],
            [-1j * Kz_top_s, O, -big_G_11, -big_G_12],
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

    final_RT = np.linalg.inv(final_A) @ final_B

    R_s = final_RT[:ff_xy, :].flatten()
    R_p = final_RT[ff_xy: 2 * ff_xy, :].flatten()

    # TODO: check why this is not applied for TE
    big_T1 = final_RT[2 * ff_xy:, :]
    big_T = big_T @ big_T1

    T_s = big_T[:ff_xy, :].flatten()
    T_p = big_T[ff_xy:, :].flatten()

    de_ri = R_s * np.conj(R_s) * np.real(np.diag(Kz_top_s) / (n_I * np.cos(theta))) \
            + R_p * np.conj(R_p) * np.real(np.diag(Kz_top_p) / (n_I * np.cos(theta)))

    de_ti = T_s * np.conj(T_s) * np.real(np.diag(Kz_bot_s) / (n_I * np.cos(theta))) \
            + T_p * np.conj(T_p) * np.real((np.diag(Kz_bot_p) / n_II ** 2) / (n_I * np.cos(theta)))


    return de_ri.real, de_ti.real, big_T1
