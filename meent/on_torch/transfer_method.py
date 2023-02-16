import torch

from .primitives import Eig


def transfer_1d_1(ff, polarization, k0, n_I, n_II, kx_vector, theta, delta_i0, fourier_order,
                  device='cpu', type_complex=torch.complex128):

    # kx_vector = k0 * (n_I * torch.sin(theta) - fourier_indices * (wavelength / period[0])).type(type_complex)

    k_I_z = (k0 ** 2 * n_I ** 2 - kx_vector ** 2) ** 0.5
    k_II_z = (k0 ** 2 * n_II ** 2 - kx_vector ** 2) ** 0.5

    k_I_z = torch.conj(k_I_z)
    k_II_z = torch.conj(k_II_z)

    Kx = torch.diag(kx_vector / k0)

    f = torch.eye(ff, device=device, dtype=type_complex)

    if polarization == 0:  # TE
        Y_I = torch.diag(k_I_z / k0)
        Y_II = torch.diag(k_II_z / k0)

        YZ_I = Y_I
        g = 1j * Y_II
        inc_term = 1j * n_I * torch.cos(theta) * delta_i0

    elif polarization == 1:  # TM
        Z_I = torch.diag(k_I_z / (k0 * n_I ** 2))
        Z_II = torch.diag(k_II_z / (k0 * n_II ** 2))

        YZ_I = Z_I
        g = 1j * Z_II
        inc_term = 1j * delta_i0 * torch.cos(theta) / n_I

    else:
        raise ValueError

    T = torch.eye(2 * fourier_order + 1, device=device, dtype=type_complex)

    return kx_vector, Kx, k_I_z, k_II_z, f, YZ_I, g, inc_term, T


def transfer_1d_2(k0, q, d, W, V, f, g, fourier_order, T, device='cpu', type_complex=torch.complex128):

    X = torch.diag(torch.exp(-k0 * q * d))

    W_i = torch.linalg.inv(W)
    V_i = torch.linalg.inv(V)

    a = 0.5 * (W_i @ f + V_i @ g)
    b = 0.5 * (W_i @ f - V_i @ g)

    a_i = torch.linalg.inv(a)

    f = W @ (torch.eye(2 * fourier_order + 1, device=device, dtype=type_complex) + X @ b @ a_i @ X)
    g = V @ (torch.eye(2 * fourier_order + 1, device=device, dtype=type_complex) - X @ b @ a_i @ X)
    T = T @ a_i @ X

    return X, f, g, T, a_i, b


def transfer_1d_3(g1, YZ_I, f1, delta_i0, inc_term, T, k_I_z, k0, n_I, n_II, theta, polarization, k_II_z):

    T1 = torch.linalg.inv(g1 + 1j * YZ_I @ f1) @ (1j * YZ_I @ delta_i0 + inc_term)
    R = f1 @ T1 - delta_i0
    T = T @ T1

    de_ri = torch.real(R * torch.conj(R) * k_I_z / (k0 * n_I * torch.cos(theta)))
    if polarization == 0:
        # de_ti = T * np.conj(T) * np.real(k_II_z / (k0 * n_I * np.cos(theta)))
        de_ti = torch.real(T * torch.conj(T) * k_II_z / (k0 * n_I * torch.cos(theta)))
    elif polarization == 1:
        # de_ti = T * np.conj(T) * np.real(k_II_z / n_II ** 2) / (k0 * np.cos(theta) / n_I)
        de_ti = torch.real(T * torch.conj(T) * k_II_z / n_II ** 2) / (k0 * torch.cos(theta) / n_I)
    else:
        raise ValueError

    return de_ri, de_ti, T1


def transfer_1d_conical_1(ff, k0, n_I, n_II, kx_vector, theta, phi, device='cpu', type_complex=torch.complex128):

    I = torch.eye(ff, device=device, dtype=type_complex)
    O = torch.zeros((ff, ff), device=device, dtype=type_complex)

    # kx_vector = k0 * (n_I * torch.sin(theta) * torch.cos(phi) - fourier_indices * (
    #         wavelength / period[0])).type(type_complex)

    ky = k0 * n_I * torch.sin(theta) * torch.sin(phi)

    k_I_z = (k0 ** 2 * n_I ** 2 - kx_vector ** 2 - ky ** 2) ** 0.5
    k_II_z = (k0 ** 2 * n_II ** 2 - kx_vector ** 2 - ky ** 2) ** 0.5

    k_I_z = torch.conj(k_I_z.flatten())
    k_II_z = torch.conj(k_II_z.flatten())

    Kx = torch.diag(kx_vector / k0)

    varphi = torch.arctan(ky / kx_vector)

    Y_I = torch.diag(k_I_z / k0)
    Y_II = torch.diag(k_II_z / k0)

    Z_I = torch.diag(k_I_z / (k0 * n_I ** 2))
    Z_II = torch.diag(k_II_z / (k0 * n_II ** 2))

    big_F = torch.cat(
        [
            torch.cat([I, O], dim=1),
            torch.cat([O, 1j * Z_II], dim=1),
        ]
    )

    big_G = torch.cat(
        [
            torch.cat([1j * Y_II, O], dim=1),
            torch.cat([O, I], dim=1),
        ]
    )

    big_T = torch.eye(ff * 2, device=device, dtype=type_complex)

    return Kx, ky, k_I_z, k_II_z, varphi, Y_I, Y_II, Z_I, Z_II, big_F, big_G, big_T


def transfer_1d_conical_2(k0, Kx, ky, E_conv, E_i, o_E_conv_i, ff, d, varphi, big_F, big_G, big_T,
                          device='cpu', type_complex=torch.complex128, perturbation=1E-10):

    I = torch.eye(ff, device=device, dtype=type_complex)
    O = torch.zeros((ff, ff), device=device, dtype=type_complex)

    A = Kx ** 2 - E_conv
    B = Kx @ E_i @ Kx - I
    A_i = torch.linalg.inv(A)
    B_i = torch.linalg.inv(B)

    to_decompose_W_1 = ky ** 2 * I + A
    to_decompose_W_2 = ky ** 2 * I + B @ o_E_conv_i

    # eigenvalues_1, W_1 = torch.linalg.eig(to_decompose_W_1)
    # eigenvalues_2, W_2 = torch.linalg.eig(to_decompose_W_2)

    Eig.broadening_parameter = perturbation
    eigenvalues_1, W_1 = Eig.apply(to_decompose_W_1)
    eigenvalues_2, W_2 = Eig.apply(to_decompose_W_2)

    q_1 = eigenvalues_1 ** 0.5
    q_2 = eigenvalues_2 ** 0.5

    Q_1 = torch.diag(q_1)
    Q_2 = torch.diag(q_2)

    V_11 = A_i @ W_1 @ Q_1
    V_12 = (ky / k0) * A_i @ Kx @ W_2
    V_21 = (ky / k0) * B_i @ Kx @ E_i @ W_1
    V_22 = B_i @ W_2 @ Q_2

    X_1 = torch.diag(torch.exp(-k0 * q_1 * d))
    X_2 = torch.diag(torch.exp(-k0 * q_2 * d))

    F_c = torch.diag(torch.cos(varphi))
    F_s = torch.diag(torch.sin(varphi))

    V_ss = F_c @ V_11
    V_sp = F_c @ V_12 - F_s @ W_2
    W_ss = F_c @ W_1 + F_s @ V_21
    W_sp = F_s @ V_22
    W_ps = F_s @ V_11
    W_pp = F_c @ W_2 + F_s @ V_12
    V_ps = F_c @ V_21 - F_s @ W_1
    V_pp = F_c @ V_22

    big_I = torch.eye(2 * (len(I)), device=device, dtype=type_complex)

    big_X = torch.cat([
        torch.cat([X_1, O], dim=1),
        torch.cat([O, X_2], dim=1)])

    big_W = torch.cat([
        torch.cat([V_ss, V_sp], dim=1),
        torch.cat([W_ps, W_pp], dim=1)])

    big_V = torch.cat([
        torch.cat([W_ss, W_sp],  dim=1),
        torch.cat([V_ps, V_pp], dim=1)])

    big_W_i = torch.linalg.inv(big_W)
    big_V_i = torch.linalg.inv(big_V)


    big_A = 0.5 * (big_W_i @ big_F + big_V_i @ big_G)
    big_B = 0.5 * (big_W_i @ big_F - big_V_i @ big_G)

    big_A_i = torch.linalg.inv(big_A)

    big_F = big_W @ (big_I + big_X @ big_B @ big_A_i @ big_X)
    big_G = big_V @ (big_I - big_X @ big_B @ big_A_i @ big_X)

    big_T = big_T @ big_A_i @ big_X

    return big_X, big_F, big_G, big_T, big_A_i, big_B, W_1, W_2, V_11, V_12, V_21, V_22, q_1, q_2


def transfer_1d_conical_3(big_F, big_G, big_T, Z_I, Y_I, psi, theta, ff, delta_i0, k_I_z, k0, n_I, n_II, k_II_z,
                          device='cpu', type_complex=torch.complex128):

    I = torch.eye(ff, device=device, dtype=type_complex)
    O = torch.zeros((ff, ff), device=device, dtype=type_complex)

    big_F_11 = big_F[:ff, :ff]
    big_F_12 = big_F[:ff, ff:]
    big_F_21 = big_F[ff:, :ff]
    big_F_22 = big_F[ff:, ff:]

    big_G_11 = big_G[:ff, :ff]
    big_G_12 = big_G[:ff, ff:]
    big_G_21 = big_G[ff:, :ff]
    big_G_22 = big_G[ff:, ff:]

    final_A = torch.cat(
        [
            torch.cat([I, O, -big_F_11, -big_F_12], dim=1),
            torch.cat([O, -1j * Z_I, -big_F_21, -big_F_22], dim=1),
            torch.cat([-1j * Y_I, O, -big_G_11, -big_G_12], dim=1),
            torch.cat([O, I, -big_G_21, -big_G_22], dim=1),
        ]
    )

    final_B = torch.cat([
        -torch.sin(psi) * delta_i0,
        -torch.cos(psi) * torch.cos(theta) * delta_i0,
        -1j * torch.sin(psi) * n_I * torch.cos(theta) * delta_i0,
        1j * n_I * torch.cos(psi) * delta_i0
    ])

    final_RT = torch.linalg.inv(final_A) @ final_B

    R_s = final_RT[:ff].flatten()
    R_p = final_RT[ff:2 * ff].flatten()

    big_T1 = final_RT[2 * ff:]
    big_T = big_T @ big_T1

    T_s = big_T[:ff].flatten()
    T_p = big_T[ff:].flatten()

    de_ri = R_s * torch.conj(R_s) * torch.real(k_I_z / (k0 * n_I * torch.cos(theta))) \
           + R_p * torch.conj(R_p) * torch.real((k_I_z / n_I ** 2) / (k0 * n_I * torch.cos(theta)))

    de_ti = T_s * torch.conj(T_s) * torch.real(k_II_z / (k0 * n_I * torch.cos(theta))) \
           + T_p * torch.conj(T_p) * torch.real((k_II_z / n_II ** 2) / (k0 * n_I * torch.cos(theta)))

    return de_ri.real, de_ti.real, big_T1



def transfer_2d_1(ff, k0, n_I, n_II, kx_vector, period, fourier_indices, theta, phi, wavelength,
                  device='cpu', type_complex=torch.complex128):

    I = torch.eye(ff ** 2, device=device, dtype=type_complex)
    O = torch.zeros((ff ** 2, ff ** 2), device=device, dtype=type_complex)

    # kx_vector = k0 * (n_I * torch.sin(theta) * torch.cos(phi) - fourier_indices * (
    #         wavelength / period[0])).type(type_complex)
    ky_vector = k0 * (n_I * torch.sin(theta) * torch.sin(phi) - fourier_indices * (
            wavelength / period[1])).type(type_complex)

    k_I_z = (k0 ** 2 * n_I ** 2 - kx_vector ** 2 - ky_vector.reshape((-1, 1)) ** 2) ** 0.5
    k_II_z = (k0 ** 2 * n_II ** 2 - kx_vector ** 2 - ky_vector.reshape((-1, 1)) ** 2) ** 0.5

    k_I_z = torch.conj(k_I_z.flatten())
    k_II_z = torch.conj(k_II_z.flatten())

    Kx = torch.diag(kx_vector.tile(ff).flatten() / k0)
    Ky = torch.diag(ky_vector.reshape((-1, 1)).tile(ff).flatten() / k0)

    varphi = torch.arctan(ky_vector.reshape((-1, 1)) / kx_vector).flatten()

    Y_I = torch.diag(k_I_z / k0)
    Y_II = torch.diag(k_II_z / k0)

    Z_I = torch.diag(k_I_z / (k0 * n_I ** 2))
    Z_II = torch.diag(k_II_z / (k0 * n_II ** 2))

    big_F = torch.cat(
        [
            torch.cat([I, O], dim=1),
            torch.cat([O, 1j * Z_II], dim=1),
        ]
    )

    big_G = torch.cat(
        [
            torch.cat([1j * Y_II, O], dim=1),
            torch.cat([O, I], dim=1),
        ]
    )

    big_T = torch.eye(ff ** 2 * 2, device=device, dtype=type_complex)

    return kx_vector, ky_vector, Kx, Ky, k_I_z, k_II_z, varphi, Y_I, Y_II, Z_I, Z_II, big_F, big_G, big_T


def transfer_2d_wv(ff, Kx, E_conv_i, Ky, o_E_conv_i, E_conv, device='cpu', type_complex=torch.complex128,
                   perturbation=1E-10):

    I = torch.eye(ff ** 2, device=device, dtype=type_complex)

    B = Kx @ E_conv_i @ Kx - I
    D = Ky @ E_conv_i @ Ky - I

    S2_from_S = torch.cat(
        [
            torch.cat([Ky ** 2 + B @ o_E_conv_i, Kx @ (E_conv_i @ Ky @ E_conv - Ky)], dim=1),
            torch.cat([Ky @ (E_conv_i @ Kx @ o_E_conv_i - Kx), Kx ** 2 + D @ E_conv], dim=1)
        ])
    Eig.broadening_parameter = perturbation
    eigenvalues, W = Eig.apply(S2_from_S)

    q = eigenvalues ** 0.5

    Q = torch.diag(q)
    Q_i = torch.linalg.inv(Q)
    U1_from_S = torch.cat(
        [
            torch.cat([-Kx @ Ky, Kx ** 2 - o_E_conv_i], dim=1),
            torch.cat([o_E_conv_i - Ky ** 2, Ky @ Kx], dim=1)
        ]
    )
    V = U1_from_S @ W @ Q_i

    return W, V, q


def transfer_2d_2(k0, d, W, V, center, q, varphi, I, O, big_F, big_G, big_T, device='cpu',
                  type_complex=torch.complex128):

    q1 = q[:center]
    q2 = q[center:]

    W_11 = W[:center, :center]
    W_12 = W[:center, center:]
    W_21 = W[center:, :center]
    W_22 = W[center:, center:]

    V_11 = V[:center, :center]
    V_12 = V[:center, center:]
    V_21 = V[center:, :center]
    V_22 = V[center:, center:]

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

    return big_X, big_F, big_G, big_T, big_A_i, big_B, W_11, W_12, W_21, W_22, V_11, V_12, V_21, V_22


def transfer_2d_3(center, big_F, big_G, big_T, Z_I, Y_I, psi, theta, ff, delta_i0, k_I_z, k0, n_I, n_II, k_II_z,
                  device='cpu', type_complex=torch.complex128):

    I = torch.eye(ff ** 2, device=device, dtype=type_complex)
    O = torch.zeros((ff ** 2, ff ** 2), device=device, dtype=type_complex)

    big_F_11 = big_F[:center, :center]
    big_F_12 = big_F[:center, center:]
    big_F_21 = big_F[center:, :center]
    big_F_22 = big_F[center:, center:]

    big_G_11 = big_G[:center, :center]
    big_G_12 = big_G[:center, center:]
    big_G_21 = big_G[center:, :center]
    big_G_22 = big_G[center:, center:]

    # Final Equation in form of AX=B
    final_A = torch.cat(
        [
            torch.cat([I, O, -big_F_11, -big_F_12], dim=1),
            torch.cat([O, -1j * Z_I, -big_F_21, -big_F_22], dim=1),
            torch.cat([-1j * Y_I, O, -big_G_11, -big_G_12], dim=1),
            torch.cat([O, I, -big_G_21, -big_G_22], dim=1),
        ]
    )

    final_B = torch.cat(
        [
            torch.cat([-torch.sin(psi) * delta_i0], dim=1),
            torch.cat([-torch.cos(psi) * torch.cos(theta) * delta_i0], dim=1),
            torch.cat([-1j * torch.sin(psi) * n_I * torch.cos(theta) * delta_i0], dim=1),
            torch.cat([1j * n_I * torch.cos(psi) * delta_i0], dim=1),
        ]
    )

    final_RT = torch.linalg.inv(final_A) @ final_B

    R_s = final_RT[:ff ** 2, :].flatten()
    R_p = final_RT[ff ** 2:2 * ff ** 2, :].flatten()

    big_T1 = final_RT[2 * ff ** 2:, :]
    big_T = big_T @ big_T1

    T_s = big_T[:ff ** 2, :].flatten()
    T_p = big_T[ff ** 2:, :].flatten()

    de_ri = R_s * torch.conj(R_s) * torch.real(k_I_z / (k0 * n_I * torch.cos(theta))) \
            + R_p * torch.conj(R_p) * torch.real((k_I_z / n_I ** 2) / (k0 * n_I * torch.cos(theta)))

    de_ti = T_s * torch.conj(T_s) * torch.real(k_II_z / (k0 * n_I * torch.cos(theta))) \
            + T_p * torch.conj(T_p) * torch.real((k_II_z / n_II ** 2) / (k0 * n_I * torch.cos(theta)))

    return de_ri.real, de_ti.real, big_T1
