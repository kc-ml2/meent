import torch

from .primitives import Eig


def transfer_1d_1(pol, ff_x, kx, n_top, n_bot, device=torch.device('cpu'), type_complex=torch.complex128):

    ff_xy = ff_x * 1

    kz_top = (n_top ** 2 - kx ** 2) ** 0.5
    kz_bot = (n_bot ** 2 - kx ** 2) ** 0.5
    kz_top = torch.conj(kz_top)
    kz_bot = torch.conj(kz_bot)

    F = torch.eye(ff_xy, device=device, dtype=type_complex)

    if pol == 0:  # TE
        Kz_bot = torch.diag(kz_bot)
        G = 1j * Kz_bot
    elif pol == 1:  # TM
        Kz_bot = torch.diag(kz_bot / (n_bot ** 2))
        G = 1j * Kz_bot
    else:
        raise ValueError

    T = torch.eye(ff_xy, device=device, dtype=type_complex)

    return kz_top, kz_bot, F, G, T


def transfer_1d_2(pol, kx, epx_conv, epy_conv, epz_conv_i, device=torch.device('cpu'), type_complex=torch.complex128, perturbation=1E-10):

    Kx = torch.diag(kx)

    if pol == 0:
        A = Kx ** 2 - epy_conv

        Eig.perturbation = perturbation
        eigenvalues, W = Eig.apply(A)

        # eigenvalues += 0j  # to get positive square root
        q = eigenvalues ** 0.5
        Q = torch.diag(q)
        V = W @ Q

    elif pol == 1:
        B = Kx @ epz_conv_i @ Kx - torch.eye(epy_conv.shape[0], device=device, dtype=type_complex)

        Eig.perturbation = perturbation
        eigenvalues, W = Eig.apply(epx_conv @ B)

        # eigenvalues += 0j  # to get positive square root
        q = eigenvalues ** 0.5

        Q = torch.diag(q)
        V = torch.linalg.inv(epx_conv) @ W @ Q

    else:
        raise ValueError

    return W, V, q


def transfer_1d_2_(k0, q, d, W, V, f, g, fourier_order, T, device=torch.device('cpu'), type_complex=torch.complex128):

    X = torch.diag(torch.exp(-k0 * q * d))

    W_i = torch.linalg.inv(W)
    V_i = torch.linalg.inv(V)

    a = 0.5 * (W_i @ f + V_i @ g)
    b = 0.5 * (W_i @ f - V_i @ g)

    a_i = torch.linalg.inv(a)

    f = W @ (torch.eye(2 * fourier_order[0] + 1, device=device, dtype=type_complex) + X @ b @ a_i @ X)
    g = V @ (torch.eye(2 * fourier_order[0] + 1, device=device, dtype=type_complex) - X @ b @ a_i @ X)
    T = T @ a_i @ X

    return X, f, g, T, a_i, b


def transfer_1d_3(k0, W, V, q, d, F, G, T, device=torch.device('cpu'), type_complex=torch.complex128):

    ff_x = len(q)

    I = torch.eye(ff_x, device=device, dtype=type_complex)

    X = torch.diag(torch.exp(-k0 * q * d))

    W_i = torch.linalg.inv(W)
    V_i = torch.linalg.inv(V)

    A = 0.5 * (W_i @ F + V_i @ G)
    B = 0.5 * (W_i @ F - V_i @ G)

    A_i = torch.linalg.inv(A)

    F = W @ (I + X @ B @ A_i @ X)
    G = V @ (I - X @ B @ A_i @ X)
    T = T @ A_i @ X

    return X, F, G, T, A_i, B


def transfer_1d_4(pol, F, G, T, kz_top, kz_bot, theta, n_top, n_bot, device=torch.device('cpu'), type_complex=torch.complex128):

    ff_xy = len(kz_top)

    Kz_top = torch.diag(kz_top)

    delta_i0 = torch.zeros(ff_xy, device=device, dtype=type_complex)
    delta_i0[ff_xy // 2] = 1

    if pol == 0:  # TE
        inc_term = 1j * n_top * torch.cos(theta) * delta_i0
        T1 = torch.linalg.inv(G + 1j * Kz_top @ F) @ (1j * Kz_top @ delta_i0 + inc_term)

    elif pol == 1:  # TM
        inc_term = 1j * delta_i0 * torch.cos(theta) / n_top
        T1 = torch.linalg.inv(G + 1j * Kz_top / (n_top ** 2) @ F) @ (1j * Kz_top / (n_top ** 2) @ delta_i0 + inc_term)

    # T1 = np.linalg.inv(G + 1j * YZ_I @ F) @ (1j * YZ_I @ delta_i0 + inc_term)
    R = F @ T1 - delta_i0
    T = T @ T1

    de_ri = torch.real(R * torch.conj(R) * kz_top / (n_top * torch.cos(theta)))

    if pol == 0:
        de_ti = T * torch.conj(T) * torch.real(kz_bot / (n_top * torch.cos(theta)))
    elif pol == 1:
        de_ti = T * torch.conj(T) * torch.real(kz_bot / n_bot ** 2) / (torch.cos(theta) / n_top)
    else:
        raise ValueError

    return de_ri.real, de_ti.real, T1, [R], [T]


def transfer_2d_1(ff_x, ff_y, kx, ky, n_top, n_bot, device=torch.device('cpu'), type_complex=torch.complex128):

    ff_xy = ff_x * ff_y

    I = torch.eye(ff_xy, device=device, dtype=type_complex)
    O = torch.zeros((ff_xy, ff_xy), device=device, dtype=type_complex)

    kz_top = (n_top ** 2 - kx ** 2 - ky.reshape((-1, 1)) ** 2) ** 0.5
    kz_bot = (n_bot ** 2 - kx ** 2 - ky.reshape((-1, 1)) ** 2) ** 0.5

    kz_top = torch.conj(kz_top).flatten()
    kz_bot = torch.conj(kz_bot).flatten()

    varphi = torch.arctan(ky.reshape((-1, 1)) / kx).flatten()

    Kz_bot = torch.diag(kz_bot)

    big_F = torch.cat(
        [
            torch.cat([I, O], dim=1),
            torch.cat([O, 1j * Kz_bot / (n_bot ** 2)], dim=1),
        ]
    )

    big_G = torch.cat(
        [
            torch.cat([1j * Kz_bot, O], dim=1),
            torch.cat([O, I], dim=1),
        ]
    )

    big_T = torch.eye(2 * ff_xy, device=device, dtype=type_complex)

    return kz_top, kz_bot, varphi, big_F, big_G, big_T


def transfer_2d_2(kx, ky, epx_conv, epy_conv, epz_conv_i, device=torch.device('cpu'), type_complex=torch.complex128,
                   perturbation=1E-10):

    ff_x = len(kx)
    ff_y = len(ky)
    ff_xy = ff_x * ff_y

    # I = np.eye(ff_y * ff_x, dtype=type_complex)
    I = torch.eye(ff_xy, device=device, dtype=type_complex)

    # Kx = torch.diag(torch.tile(kx, ff_y).flatten())
    # Ky = torch.diag(torch.tile(ky.reshape((-1, 1)), ff_x).flatten())

    Kx = torch.diag(kx.tile(ff_y).flatten())
    Ky = torch.diag(ky.reshape((-1, 1)).tile(ff_x).flatten())

    B = Kx @ epz_conv_i @ Kx - I
    D = Ky @ epz_conv_i @ Ky - I

    Omega2_LR = torch.cat(
        [
            torch.cat([Ky ** 2 + B @ epx_conv, Kx @ (epz_conv_i @ Ky @ epy_conv - Ky)], dim=1),
            torch.cat([Ky @ (epz_conv_i @ Kx @ epx_conv - Kx), Kx ** 2 + D @ epy_conv], dim=1)
        ])

    Eig.perturbation = perturbation
    eigenvalues, W = Eig.apply(Omega2_LR)
    q = eigenvalues ** 0.5

    Q = torch.diag(q)
    Q_i = torch.linalg.inv(Q)
    Omega_R = torch.cat(
        [
            torch.cat([-Kx @ Ky, Kx ** 2 - epy_conv], dim=1),
            torch.cat([epx_conv - Ky ** 2, Ky @ Kx], dim=1)
        ]
    )
    V = Omega_R @ W @ Q_i

    return W, V, q


def transfer_2d_3(k0, W, V, q, d, varphi, big_F, big_G, big_T, device=torch.device('cpu'), type_complex=torch.complex128):
    ff_xy = len(q)//2

    I = torch.eye(ff_xy, device=device, dtype=type_complex)
    O = torch.zeros((ff_xy, ff_xy), device=device, dtype=type_complex)

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

    X_1 = torch.diag(torch.exp(-k0 * q1 * d))
    X_2 = torch.diag(torch.exp(-k0 * q2 * d))

    F_c = torch.diag(torch.cos(varphi))
    F_s = torch.diag(torch.sin(varphi))

    W_ss = F_c @ W_21 - F_s @ W_11
    W_sp = F_c @ W_22 - F_s @ W_12
    W_ps = F_c @ W_11 + F_s @ W_21
    W_pp = F_c @ W_12 + F_s @ W_22

    V_ss = F_c @ V_11 + F_s @ V_21
    V_sp = F_c @ V_12 + F_s @ V_22
    V_ps = F_c @ V_21 - F_s @ V_11
    V_pp = F_c @ V_22 - F_s @ V_12

    big_I = torch.eye(2 * (len(I)), device=device, dtype=type_complex)

    big_X = torch.cat([
        torch.cat([X_1, O], dim=1),
        torch.cat([O, X_2], dim=1)])

    big_W = torch.cat([
        torch.cat([W_ss, W_sp], dim=1),
        torch.cat([W_ps, W_pp], dim=1)])

    big_V = torch.cat([
        torch.cat([V_ss, V_sp],  dim=1),
        torch.cat([V_ps, V_pp], dim=1)])

    big_W_i = torch.linalg.inv(big_W)
    big_V_i = torch.linalg.inv(big_V)

    big_A = 0.5 * (big_W_i @ big_F + big_V_i @ big_G)
    big_B = 0.5 * (big_W_i @ big_F - big_V_i @ big_G)

    big_A_i = torch.linalg.inv(big_A)

    big_F = big_W @ (big_I + big_X @ big_B @ big_A_i @ big_X)
    big_G = big_V @ (big_I - big_X @ big_B @ big_A_i @ big_X)

    big_T = big_T @ big_A_i @ big_X

    return big_X, big_F, big_G, big_T, big_A_i, big_B


def transfer_2d_4(big_F, big_G, big_T, kz_top, kz_bot, psi, theta, n_top, n_bot,
                  device=torch.device('cpu'), type_complex=torch.complex128):

    ff_xy = len(big_F) // 2

    Kz_top = torch.diag(kz_top)

    I = torch.eye(ff_xy, device=device, dtype=type_complex)
    O = torch.zeros((ff_xy, ff_xy), device=device, dtype=type_complex)

    big_F_11 = big_F[:ff_xy, :ff_xy]
    big_F_12 = big_F[:ff_xy, ff_xy:]
    big_F_21 = big_F[ff_xy:, :ff_xy]
    big_F_22 = big_F[ff_xy:, ff_xy:]

    big_G_11 = big_G[:ff_xy, :ff_xy]
    big_G_12 = big_G[:ff_xy, ff_xy:]
    big_G_21 = big_G[ff_xy:, :ff_xy]
    big_G_22 = big_G[ff_xy:, ff_xy:]

    delta_i0 = torch.zeros((ff_xy, 1), device=device, dtype=type_complex)
    delta_i0[ff_xy // 2, 0] = 1

    # Final Equation in form of AX=B
    final_A = torch.cat(
        [
            torch.cat([I, O, -big_F_11, -big_F_12], dim=1),
            torch.cat([O, -1j * Kz_top / (n_top ** 2), -big_F_21, -big_F_22], dim=1),
            torch.cat([-1j * Kz_top, O, -big_G_11, -big_G_12], dim=1),
            torch.cat([O, I, -big_G_21, -big_G_22], dim=1),
        ]
    )

    final_B = torch.cat(
        [
            torch.cat([-torch.sin(psi) * delta_i0], dim=1),
            torch.cat([torch.cos(psi) * torch.cos(theta) * delta_i0], dim=1),
            torch.cat([-1j * torch.sin(psi) * n_top * torch.cos(theta) * delta_i0], dim=1),
            torch.cat([-1j * n_top * torch.cos(psi) * delta_i0], dim=1),
        ]
    )

    final_RT = torch.linalg.inv(final_A) @ final_B

    R_s = final_RT[:ff_xy, :].flatten()  # TODO: why flatten?
    R_p = final_RT[ff_xy:2 * ff_xy, :].flatten()

    big_T1 = final_RT[2 * ff_xy:, :]
    big_T = big_T @ big_T1

    T_s = big_T[:ff_xy, :].flatten()
    T_p = big_T[ff_xy:, :].flatten()

    de_ri = R_s * torch.conj(R_s) * torch.real(kz_top / (n_top * torch.cos(theta))) \
            + R_p * torch.conj(R_p) * torch.real((kz_top / n_top ** 2) / (n_top * torch.cos(theta)))

    de_ti = T_s * torch.conj(T_s) * torch.real(kz_bot / (n_top * torch.cos(theta))) \
            + T_p * torch.conj(T_p) * torch.real((kz_bot / n_bot ** 2) / (n_top * torch.cos(theta)))

    return de_ri.real, de_ti.real, big_T1, [R_s, R_p], [T_s, T_p]
