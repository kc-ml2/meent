import jax.numpy as jnp

# from .convolution_matrix import *


def transfer_1d_1(ff, polarization, k0, n_I, n_II, theta, delta_i0, fourier_order,fourier_indices, wl, period):

    kx_vector = k0 * (n_I * jnp.sin(theta) - fourier_indices * (wl / period[0])).astype('complex')

    k_I_z = (k0 ** 2 * n_I ** 2 - kx_vector ** 2) ** 0.5
    k_II_z = (k0 ** 2 * n_II ** 2 - kx_vector ** 2) ** 0.5

    k_I_z = k_I_z.conjugate()
    k_II_z = k_II_z.conjugate()

    Kx = jnp.diag(kx_vector / k0)

    f = jnp.eye(ff)

    if polarization == 0:  # TE
        Y_I = jnp.diag(k_I_z / k0)
        Y_II = jnp.diag(k_II_z / k0)

        YZ_I = Y_I
        g = 1j * Y_II
        inc_term = 1j * n_I * jnp.cos(theta) * delta_i0

    elif polarization == 1:  # TM
        Z_I = jnp.diag(k_I_z / (k0 * n_I ** 2))
        Z_II = jnp.diag(k_II_z / (k0 * n_II ** 2))

        YZ_I = Z_I
        g = 1j * Z_II
        inc_term = 1j * delta_i0 * jnp.cos(theta) / n_I

    else:
        raise ValueError

    T = jnp.eye(2 * fourier_order + 1)

    return Kx, k_I_z, k_II_z, Kx, f, YZ_I, g, inc_term, T


def transfer_1d_2(k0, q, d, W, V, f, g, fourier_order, T):
    X = jnp.diag(jnp.exp(-k0 * q * d))

    W_i = jnp.linalg.inv(W)
    V_i = jnp.linalg.inv(V)

    a = 0.5 * (W_i @ f + V_i @ g)
    b = 0.5 * (W_i @ f - V_i @ g)

    a_i = jnp.linalg.inv(a)

    f = W @ (jnp.eye(2 * fourier_order + 1) + X @ b @ a_i @ X)
    g = V @ (jnp.eye(2 * fourier_order + 1) - X @ b @ a_i @ X)
    T = T @ a_i @ X

    return f, g, T


def transfer_1d_3(g, YZ_I, f, delta_i0, inc_term, T, k_I_z, k0, n_I, n_II, theta, polarization, k_II_z):
    Tl = jnp.linalg.inv(g + 1j * YZ_I @ f) @ (1j * YZ_I @ delta_i0 + inc_term)
    R = f @ Tl - delta_i0
    T = T @ Tl

    de_ri = jnp.real(R * jnp.conj(R) * k_I_z / (k0 * n_I * jnp.cos(theta)))
    if polarization == 0:
        # de_ti = T * jnp.conj(T) * jnp.real(k_II_z / (k0 * n_I * jnp.cos(theta)))
        de_ti = jnp.real(T * jnp.conj(T) * k_II_z / (k0 * n_I * jnp.cos(theta)))
    elif polarization == 1:
        # de_ti = T * jnp.conj(T) * jnp.real(k_II_z / n_II ** 2) / (k0 * jnp.cos(theta) / n_I)
        de_ti = jnp.real(T * jnp.conj(T) * k_II_z / n_II ** 2) / (k0 * jnp.cos(theta) / n_I)
    else:
        raise ValueError

    return de_ri, de_ti


def transfer_2d_1(ff, k0, n_I, n_II, period, fourier_indices, theta, phi, wl, perturbation=1E-20*(1+1j)):
    I = jnp.eye(ff ** 2)
    O = jnp.zeros((ff ** 2, ff ** 2))

    kx_vector = k0 * (n_I * jnp.sin(theta) * jnp.cos(phi) - fourier_indices * (
            wl / period[0])).astype('complex')
    ky_vector = k0 * (n_I * jnp.sin(theta) * jnp.sin(phi) - fourier_indices * (
            wl / period[1])).astype('complex')

    Kx = jnp.diag(jnp.tile(kx_vector, ff).flatten()) / k0
    Ky = jnp.diag(jnp.tile(ky_vector.reshape((-1, 1)), ff).flatten()) / k0

    k_I_z = (k0 ** 2 * n_I ** 2 - kx_vector ** 2 - ky_vector.reshape((-1, 1)) ** 2) ** 0.5
    k_II_z = (k0 ** 2 * n_II ** 2 - kx_vector ** 2 - ky_vector.reshape((-1, 1)) ** 2) ** 0.5

    k_I_z = k_I_z.flatten().conjugate()
    k_II_z = k_II_z.flatten().conjugate()

    idx = jnp.nonzero(kx_vector == 0)[0]
    if len(idx):
        # TODO: need imaginary part?
        # TODO: make imaginary part sign consistent
        kx_vector = kx_vector.at[idx].set(perturbation)
        print(wl, 'varphi divide by 0: adding perturbation')

    varphi = jnp.arctan(ky_vector.reshape((-1, 1)) / kx_vector).flatten()

    Y_I = jnp.diag(k_I_z / k0)
    Y_II = jnp.diag(k_II_z / k0)

    Z_I = jnp.diag(k_I_z / (k0 * n_I ** 2))
    Z_II = jnp.diag(k_II_z / (k0 * n_II ** 2))

    big_F = jnp.block([[I, O], [O, 1j * Z_II]])
    big_G = jnp.block([[1j * Y_II, O], [O, I]])

    big_T = jnp.eye(ff ** 2 * 2)

    return Kx, Ky, k_I_z, k_II_z, varphi, Y_I, Y_II, Z_I, Z_II, big_F, big_G, big_T


def transfer_2d_wv(ff, Kx, E_i, Ky, oneover_E_conv_i, E_conv, center):

    I = jnp.eye(ff ** 2)
    O = jnp.zeros((ff ** 2, ff ** 2))

    B = Kx @ E_i @ Kx - I
    D = Ky @ E_i @ Ky - I

    S2_from_S = jnp.block(
        [
            [Ky ** 2 + B @ oneover_E_conv_i, Kx @ (E_i @ Ky @ E_conv - Ky)],
            [Ky @ (E_i @ Kx @ oneover_E_conv_i - Kx), Kx ** 2 + D @ E_conv]
        ])

    eigenvalues, W = jnp.linalg.eig(S2_from_S)

    Lambda = eigenvalues ** 0.5

    # Lambda_1 = Lambda[:center]
    # Lambda_2 = Lambda[center:]

    LAMBDA = jnp.diag(Lambda)
    LAMBDA_i = jnp.linalg.inv(LAMBDA)
    U1_from_S = jnp.block(
        [
            [-Kx @ Ky, Kx ** 2 - E_conv],
            [oneover_E_conv_i - Ky ** 2, Ky @ Kx]  # TODO Check x y order
        ]
    )
    V = U1_from_S @ W @ LAMBDA_i

    return W, V, LAMBDA, Lambda


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

    X_1 = jnp.diag(jnp.exp(-k0 * Lambda_1 * d))
    X_2 = jnp.diag(jnp.exp(-k0 * Lambda_2 * d))

    F_c = jnp.diag(jnp.cos(varphi))
    F_s = jnp.diag(jnp.sin(varphi))

    W_ss = F_c @ W_21 - F_s @ W_11
    W_sp = F_c @ W_22 - F_s @ W_12
    W_ps = F_c @ W_11 + F_s @ W_21
    W_pp = F_c @ W_12 + F_s @ W_22

    V_ss = F_c @ V_11 + F_s @ V_21
    V_sp = F_c @ V_12 + F_s @ V_22
    V_ps = F_c @ V_21 - F_s @ V_11
    V_pp = F_c @ V_22 - F_s @ V_12

    big_I = jnp.eye(2 * (len(I)))
    big_X = jnp.block([[X_1, O], [O, X_2]])
    big_W = jnp.block([[W_ss, W_sp], [W_ps, W_pp]])
    big_V = jnp.block([[V_ss, V_sp], [V_ps, V_pp]])

    big_W_i = jnp.linalg.inv(big_W)
    big_V_i = jnp.linalg.inv(big_V)

    big_A = 0.5 * (big_W_i @ big_F + big_V_i @ big_G)
    big_B = 0.5 * (big_W_i @ big_F - big_V_i @ big_G)

    big_A_i = jnp.linalg.inv(big_A)

    big_F = big_W @ (big_I + big_X @ big_B @ big_A_i @ big_X)
    big_G = big_V @ (big_I - big_X @ big_B @ big_A_i @ big_X)

    big_T = big_T @ big_A_i @ big_X

    return big_F, big_G, big_T


def transfer_2d_3(center, big_F, big_G, big_T, Z_I, Y_I, psi, theta, ff, delta_i0, k_I_z, k0, n_I, n_II, k_II_z):
    I = jnp.eye(ff ** 2)
    O = jnp.zeros((ff ** 2, ff ** 2))

    big_F_11 = big_F[:center, :center]
    big_F_12 = big_F[:center, center:]
    big_F_21 = big_F[center:, :center]
    big_F_22 = big_F[center:, center:]

    big_G_11 = big_G[:center, :center]
    big_G_12 = big_G[:center, center:]
    big_G_21 = big_G[center:, :center]
    big_G_22 = big_G[center:, center:]

    # Final Equation in form of AX=B
    final_A = jnp.block(
        [
            [I, O, -big_F_11, -big_F_12],
            [O, -1j * Z_I, -big_F_21, -big_F_22],
            [-1j * Y_I, O, -big_G_11, -big_G_12],
            [O, I, -big_G_21, -big_G_22],
        ]
    )

    final_B = jnp.block(
        [
            [-jnp.sin(psi) * delta_i0],
            [-jnp.cos(psi) * jnp.cos(theta) * delta_i0],
            [-1j * jnp.sin(psi) * n_I * jnp.cos(theta) * delta_i0],
            [1j * n_I * jnp.cos(psi) * delta_i0]
        ]
    )

    final_X = jnp.linalg.inv(final_A) @ final_B

    R_s = final_X[:ff ** 2, :].flatten()
    R_p = final_X[ff ** 2:2 * ff ** 2, :].flatten()

    big_T = big_T @ final_X[2 * ff ** 2:, :]
    T_s = big_T[:ff ** 2, :].flatten()
    T_p = big_T[ff ** 2:, :].flatten()

    de_ri = R_s * jnp.conj(R_s) * jnp.real(k_I_z / (k0 * n_I * jnp.cos(theta))) \
            + R_p * jnp.conj(R_p) * jnp.real((k_I_z / n_I ** 2) / (k0 * n_I * jnp.cos(theta)))

    de_ti = T_s * jnp.conj(T_s) * jnp.real(k_II_z / (k0 * n_I * jnp.cos(theta))) \
            + T_p * jnp.conj(T_p) * jnp.real((k_II_z / n_II ** 2) / (k0 * n_I * jnp.cos(theta)))

    # Aa = de_ri.sum()
    # Aaa = de_ti.sum()
    #
    # if Aa + Aaa != 1:
    #     # TODO: no problem? or should be handled?
    #     print(1)
    #     wavelength = 1463.6363636363637
    #     deri = 350
    #
    #     wavelength = 1978.9715332727274
    #     deri = 558

    return de_ri.real, de_ti.real


def transfer_1d_conical_1(ff, k0, n_I, n_II, period, fourier_indices, theta, phi, wl, perturbation=1E-20*(1+1j)):
    I = jnp.eye(ff)
    O = jnp.zeros((ff, ff))

    kx_vector = k0 * (n_I * jnp.sin(theta) * jnp.cos(phi) - fourier_indices * (wl / period[0])).astype(
        'complex')
    ky = k0 * n_I * jnp.sin(theta) * jnp.sin(phi)

    k_I_z = (k0 ** 2 * n_I ** 2 - kx_vector ** 2 - ky ** 2) ** 0.5
    k_II_z = (k0 ** 2 * n_II ** 2 - kx_vector ** 2 - ky ** 2) ** 0.5

    k_I_z = k_I_z.conjugate()
    k_II_z = k_II_z.conjugate()

    idx = jnp.nonzero(kx_vector == 0)[0]
    if len(idx):
        # TODO: need imaginary part?
        # TODO: make imaginary part sign consistent
        kx_vector = kx_vector.at[idx].set(perturbation)
        print(wl, 'varphi divide by 0: adding perturbation')

    varphi = jnp.arctan(ky / kx_vector)

    Y_I = jnp.diag(k_I_z / k0)
    Y_II = jnp.diag(k_II_z / k0)

    Z_I = jnp.diag(k_I_z / (k0 * n_I ** 2))
    Z_II = jnp.diag(k_II_z / (k0 * n_II ** 2))

    Kx = jnp.diag(kx_vector / k0)

    big_F = jnp.block([[I, O], [O, 1j * Z_II]])
    big_G = jnp.block([[1j * Y_II, O], [O, I]])

    big_T = jnp.eye(2 * ff)

    return Kx, ky, k_I_z, k_II_z, varphi, Y_I, Y_II, Z_I, Z_II, big_F, big_G, big_T


def transfer_1d_conical_2(k0, Kx, ky, E_conv, E_i, oneover_E_conv_i, ff, d, varphi, big_F, big_G, big_T):

    I = jnp.eye(ff)
    O = jnp.zeros((ff, ff))

    A = Kx ** 2 - E_conv
    B = Kx @ E_i @ Kx - I
    A_i = jnp.linalg.inv(A)
    B_i = jnp.linalg.inv(B)

    to_decompose_W_1 = ky ** 2 * I + A
    to_decompose_W_2 = ky ** 2 * I + B @ oneover_E_conv_i

    # TODO: using eigh?
    eigenvalues_1, W_1 = jnp.linalg.eig(to_decompose_W_1)
    eigenvalues_2, W_2 = jnp.linalg.eig(to_decompose_W_2)

    q_1 = eigenvalues_1 ** 0.5
    q_2 = eigenvalues_2 ** 0.5

    Q_1 = jnp.diag(q_1)
    Q_2 = jnp.diag(q_2)

    V_11 = A_i @ W_1 @ Q_1
    V_12 = (ky / k0) * A_i @ Kx @ W_2
    V_21 = (ky / k0) * B_i @ Kx @ E_i @ W_1
    V_22 = B_i @ W_2 @ Q_2

    X_1 = jnp.diag(jnp.exp(-k0 * q_1 * d))
    X_2 = jnp.diag(jnp.exp(-k0 * q_2 * d))

    F_c = jnp.diag(jnp.cos(varphi))
    F_s = jnp.diag(jnp.sin(varphi))

    V_ss = F_c @ V_11
    V_sp = F_c @ V_12 - F_s @ W_2
    W_ss = F_c @ W_1 + F_s @ V_21
    W_sp = F_s @ V_22
    W_ps = F_s @ V_11
    W_pp = F_c @ W_2 + F_s @ V_12
    V_ps = F_c @ V_21 - F_s @ W_1
    V_pp = F_c @ V_22

    big_I = jnp.eye(2 * (len(I)))
    big_X = jnp.block([[X_1, O], [O, X_2]])
    big_W = jnp.block([[V_ss, V_sp], [W_ps, W_pp]])
    big_V = jnp.block([[W_ss, W_sp], [V_ps, V_pp]])

    big_W_i = jnp.linalg.inv(big_W)
    big_V_i = jnp.linalg.inv(big_V)

    big_A = 0.5 * (big_W_i @ big_F + big_V_i @ big_G)
    big_B = 0.5 * (big_W_i @ big_F - big_V_i @ big_G)

    big_A_i = jnp.linalg.inv(big_A)

    big_F = big_W @ (big_I + big_X @ big_B @ big_A_i @ big_X)
    big_G = big_V @ (big_I - big_X @ big_B @ big_A_i @ big_X)

    big_T = big_T @ big_A_i @ big_X

    return big_F, big_G, big_T


def transfer_1d_conical_3(big_F, big_G, big_T, Z_I, Y_I, psi, theta, ff, delta_i0, k_I_z, k0, n_I, n_II, k_II_z):
    I = jnp.eye(ff)
    O = jnp.zeros((ff, ff))

    big_F_11 = big_F[:ff, :ff]
    big_F_12 = big_F[:ff, ff:]
    big_F_21 = big_F[ff:, :ff]
    big_F_22 = big_F[ff:, ff:]

    big_G_11 = big_G[:ff, :ff]
    big_G_12 = big_G[:ff, ff:]
    big_G_21 = big_G[ff:, :ff]
    big_G_22 = big_G[ff:, ff:]

    # Final Equation in form of AX=B
    final_A = jnp.block(
        [
            [I, O, -big_F_11, -big_F_12],
            [O, -1j * Z_I, -big_F_21, -big_F_22],
            [-1j * Y_I, O, -big_G_11, -big_G_12],
            [O, I, -big_G_21, -big_G_22],
        ]
    )

    final_B = jnp.hstack([
        [-jnp.sin(psi) * delta_i0],
        [-jnp.cos(psi) * jnp.cos(theta) * delta_i0],
        [-1j * jnp.sin(psi) * n_I * jnp.cos(theta) * delta_i0],
        [1j * n_I * jnp.cos(psi) * delta_i0]
    ]).T

    final_X = jnp.linalg.inv(final_A) @ final_B

    R_s = final_X[:ff, :].flatten()
    R_p = final_X[ff:2 * ff, :].flatten()

    big_T = big_T @ final_X[2 * ff:, :]
    T_s = big_T[:ff, :].flatten()
    T_p = big_T[ff:, :].flatten()

    de_ri = R_s * jnp.conj(R_s) * jnp.real(k_I_z / (k0 * n_I * jnp.cos(theta))) \
            + R_p * jnp.conj(R_p) * jnp.real((k_I_z / n_I ** 2) / (k0 * n_I * jnp.cos(theta)))

    de_ti = T_s * jnp.conj(T_s) * jnp.real(k_II_z / (k0 * n_I * jnp.cos(theta))) \
            + T_p * jnp.conj(T_p) * jnp.real((k_II_z / n_II ** 2) / (k0 * n_I * jnp.cos(theta)))

    return de_ri.real, de_ti.real

