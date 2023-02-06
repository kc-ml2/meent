import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import pure_callback

# import meent.on_jax.jitted as ee
from . import jitted as ee
from .primitives import eig


def transfer_1d_1(ff, polarization, k0, n_I, n_II, kx_vector, theta, delta_i0, fourier_order,
                  type_complex=jnp.complex128):

    # kx_vector = k0 * (n_I * ee.sin(theta) - fourier_indices * (wavelength / period[0])).astype(type_complex)

    k_I_z = (k0 ** 2 * n_I ** 2 - kx_vector ** 2) ** 0.5
    k_II_z = (k0 ** 2 * n_II ** 2 - kx_vector ** 2) ** 0.5

    k_I_z = k_I_z.conjugate()
    k_II_z = k_II_z.conjugate()

    Kx = ee.diag(kx_vector / k0)

    f = ee.eye(ff).astype(type_complex)

    if polarization == 0:  # TE
        Y_I = ee.diag(k_I_z / k0)
        Y_II = ee.diag(k_II_z / k0)

        YZ_I = Y_I
        g = 1j * Y_II
        inc_term = 1j * n_I * ee.cos(theta) * delta_i0

    elif polarization == 1:  # TM
        Z_I = ee.diag(k_I_z / (k0 * n_I ** 2))
        Z_II = ee.diag(k_II_z / (k0 * n_II ** 2))

        YZ_I = Z_I
        g = 1j * Z_II
        inc_term = 1j * delta_i0 * ee.cos(theta) / n_I

    else:
        raise ValueError

    T = ee.eye(2 * fourier_order + 1).astype(type_complex)

    return kx_vector, Kx, k_I_z, k_II_z, Kx, f, YZ_I, g, inc_term, T


def transfer_1d_2(k0, q, d, W, V, f, g, fourier_order, T, type_complex=jnp.complex128):

    X = ee.diag(ee.exp(-k0 * q * d))

    W_i = ee.inv(W)
    V_i = ee.inv(V)

    a = 0.5 * (W_i @ f + V_i @ g)
    b = 0.5 * (W_i @ f - V_i @ g)

    a_i = ee.inv(a)

    f = W @ (ee.eye(2 * fourier_order + 1).astype(type_complex) + X @ b @ a_i @ X)
    g = V @ (ee.eye(2 * fourier_order + 1).astype(type_complex) - X @ b @ a_i @ X)
    T = T @ a_i @ X

    return X, f, g, T, a_i, b


def transfer_1d_3(g, YZ_I, f, delta_i0, inc_term, T, k_I_z, k0, n_I, n_II, theta, polarization, k_II_z):

    T1 = ee.inv(g + 1j * YZ_I @ f) @ (1j * YZ_I @ delta_i0 + inc_term)
    R = f @ T1 - delta_i0
    T = T @ T1

    de_ri = ee.real(R * ee.conj(R) * k_I_z / (k0 * n_I * ee.cos(theta)))
    if polarization == 0:
        # de_ti = T * ee.conj(T) * ee.real(k_II_z / (k0 * n_I * ee.cos(theta)))
        de_ti = ee.real(T * ee.conj(T) * k_II_z / (k0 * n_I * ee.cos(theta)))
    elif polarization == 1:
        # de_ti = T * ee.conj(T) * ee.real(k_II_z / n_II ** 2) / (k0 * ee.cos(theta) / n_I)
        de_ti = ee.real(T * ee.conj(T) * k_II_z / n_II ** 2) / (k0 * ee.cos(theta) / n_I)
    else:
        raise ValueError

    return de_ri, de_ti, T1


def transfer_1d_conical_1(ff, k0, n_I, n_II, kx_vector, theta, phi, type_complex=jnp.complex128):

    I = ee.eye(ff).astype(type_complex)
    O = ee.zeros((ff, ff)).astype(type_complex)

    # kx_vector = k0 * (n_I * ee.sin(theta) * ee.cos(phi) - fourier_indices * (wavelength / period[0])
    #                   ).astype(type_complex)

    ky = k0 * n_I * ee.sin(theta) * ee.sin(phi)

    k_I_z = (k0 ** 2 * n_I ** 2 - kx_vector ** 2 - ky ** 2) ** 0.5
    k_II_z = (k0 ** 2 * n_II ** 2 - kx_vector ** 2 - ky ** 2) ** 0.5

    k_I_z = k_I_z.conjugate()
    k_II_z = k_II_z.conjugate()

    Kx = ee.diag(kx_vector / k0)
    varphi = ee.arctan(ky / kx_vector)

    Y_I = ee.diag(k_I_z / k0)
    Y_II = ee.diag(k_II_z / k0)

    Z_I = ee.diag(k_I_z / (k0 * n_I ** 2))
    Z_II = ee.diag(k_II_z / (k0 * n_II ** 2))

    big_F = ee.block([[I, O], [O, 1j * Z_II]])
    big_G = ee.block([[1j * Y_II, O], [O, I]])

    big_T = ee.eye(2 * ff).astype(type_complex)

    return Kx, ky, k_I_z, k_II_z, varphi, Y_I, Y_II, Z_I, Z_II, big_F, big_G, big_T


def transfer_1d_conical_2(k0, Kx, ky, E_conv, E_conv_i, o_E_conv_i, ff, d, varphi, big_F, big_G, big_T,
                          type_complex=jnp.complex128, perturbation=1E-10):

    I = ee.eye(ff).astype(type_complex)
    O = ee.zeros((ff, ff)).astype(type_complex)

    A = Kx ** 2 - E_conv
    B = Kx @ E_conv_i @ Kx - I
    A_i = ee.inv(A)
    B_i = ee.inv(B)

    to_decompose_W_1 = ky ** 2 * I + A
    to_decompose_W_2 = ky ** 2 * I + B @ o_E_conv_i

    # TODO: separate to an independent func (like 2D case).
    eigenvalues_1, W_1 = eig(to_decompose_W_1, type_complex=type_complex, perturbation=perturbation)
    eigenvalues_2, W_2 = eig(to_decompose_W_2, type_complex=type_complex, perturbation=perturbation)

    q_1 = eigenvalues_1 ** 0.5
    q_2 = eigenvalues_2 ** 0.5

    Q_1 = ee.diag(q_1)
    Q_2 = ee.diag(q_2)

    V_11 = A_i @ W_1 @ Q_1
    V_12 = (ky / k0) * A_i @ Kx @ W_2
    V_21 = (ky / k0) * B_i @ Kx @ E_conv_i @ W_1
    V_22 = B_i @ W_2 @ Q_2

    X_1 = ee.diag(ee.exp(-k0 * q_1 * d))
    X_2 = ee.diag(ee.exp(-k0 * q_2 * d))

    F_c = ee.diag(ee.cos(varphi))
    F_s = ee.diag(ee.sin(varphi))

    V_ss = F_c @ V_11
    V_sp = F_c @ V_12 - F_s @ W_2
    W_ss = F_c @ W_1 + F_s @ V_21
    W_sp = F_s @ V_22
    W_ps = F_s @ V_11
    W_pp = F_c @ W_2 + F_s @ V_12
    V_ps = F_c @ V_21 - F_s @ W_1
    V_pp = F_c @ V_22

    big_I = ee.eye(2 * (len(I))).astype(type_complex)
    big_X = ee.block([[X_1, O], [O, X_2]])
    big_W = ee.block([[V_ss, V_sp], [W_ps, W_pp]])
    big_V = ee.block([[W_ss, W_sp], [V_ps, V_pp]])

    big_W_i = ee.inv(big_W)
    big_V_i = ee.inv(big_V)

    big_A = 0.5 * (big_W_i @ big_F + big_V_i @ big_G)
    big_B = 0.5 * (big_W_i @ big_F - big_V_i @ big_G)

    big_A_i = ee.inv(big_A)

    big_F = big_W @ (big_I + big_X @ big_B @ big_A_i @ big_X)
    big_G = big_V @ (big_I - big_X @ big_B @ big_A_i @ big_X)

    big_T = big_T @ big_A_i @ big_X

    return big_X, big_F, big_G, big_T, big_A_i, big_B, W_1, W_2, V_11, V_12, V_21, V_22, q_1, q_2


def transfer_1d_conical_3(big_F, big_G, big_T, Z_I, Y_I, psi, theta, ff, delta_i0, k_I_z, k0, n_I, n_II, k_II_z,
                          type_complex=jnp.complex128):

    I = ee.eye(ff).astype(type_complex)
    O = ee.zeros((ff, ff), dtype=type_complex)

    big_F_11 = big_F[:ff, :ff]
    big_F_12 = big_F[:ff, ff:]
    big_F_21 = big_F[ff:, :ff]
    big_F_22 = big_F[ff:, ff:]

    big_G_11 = big_G[:ff, :ff]
    big_G_12 = big_G[:ff, ff:]
    big_G_21 = big_G[ff:, :ff]
    big_G_22 = big_G[ff:, ff:]

    # Final Equation in form of AX=B
    final_A = ee.block(
        [
            [I, O, -big_F_11, -big_F_12],
            [O, -1j * Z_I, -big_F_21, -big_F_22],
            [-1j * Y_I, O, -big_G_11, -big_G_12],
            [O, I, -big_G_21, -big_G_22],
        ]
    )

    # tODO: correct?
    final_B = ee.hstack([
        [-ee.sin(psi) * delta_i0],
        [-ee.cos(psi) * ee.cos(theta) * delta_i0],
        [-1j * ee.sin(psi) * n_I * ee.cos(theta) * delta_i0],
        [1j * n_I * ee.cos(psi) * delta_i0]
    ]).T

    final_RT = ee.inv(final_A) @ final_B

    R_s = final_RT[:ff, :].flatten()
    R_p = final_RT[ff:2 * ff, :].flatten()

    big_T1 = final_RT[2 * ff:, :]
    big_T = big_T @ big_T1

    T_s = big_T[:ff, :].flatten()
    T_p = big_T[ff:, :].flatten()

    de_ri = R_s * ee.conj(R_s) * ee.real(k_I_z / (k0 * n_I * ee.cos(theta))) \
            + R_p * ee.conj(R_p) * ee.real((k_I_z / n_I ** 2) / (k0 * n_I * ee.cos(theta)))

    de_ti = T_s * ee.conj(T_s) * ee.real(k_II_z / (k0 * n_I * ee.cos(theta))) \
            + T_p * ee.conj(T_p) * ee.real((k_II_z / n_II ** 2) / (k0 * n_I * ee.cos(theta)))

    return de_ri.real, de_ti.real, big_T1


def transfer_2d_1(ff, k0, n_I, n_II, kx_vector, period, fourier_indices, theta, phi, wavelength,
                  type_complex=jnp.complex128):

    I = ee.eye(ff ** 2).astype(type_complex)
    O = ee.zeros((ff ** 2, ff ** 2), dtype=type_complex)

    # kx_vector = k0 * (n_I * ee.sin(theta) * ee.cos(phi) - fourier_indices * (
    #         wavelength / period[0])).astype(type_complex)

    ky_vector = k0 * (n_I * ee.sin(theta) * ee.sin(phi) - fourier_indices * (
            wavelength / period[1])).astype(type_complex)

    k_I_z = (k0 ** 2 * n_I ** 2 - kx_vector ** 2 - ky_vector.reshape((-1, 1)) ** 2) ** 0.5
    k_II_z = (k0 ** 2 * n_II ** 2 - kx_vector ** 2 - ky_vector.reshape((-1, 1)) ** 2) ** 0.5

    k_I_z = k_I_z.flatten().conjugate()
    k_II_z = k_II_z.flatten().conjugate()

    Kx = ee.diag(ee.tile(kx_vector, ff).flatten()) / k0
    Ky = ee.diag(ee.tile(ky_vector.reshape((-1, 1)), ff).flatten()) / k0

    varphi = ee.arctan(ky_vector.reshape((-1, 1)) / kx_vector).flatten()

    Y_I = ee.diag(k_I_z / k0)
    Y_II = ee.diag(k_II_z / k0)

    Z_I = ee.diag(k_I_z / (k0 * n_I ** 2))
    Z_II = ee.diag(k_II_z / (k0 * n_II ** 2))

    big_F = ee.block([[I, O], [O, 1j * Z_II]])
    big_G = ee.block([[1j * Y_II, O], [O, I]])

    big_T = ee.eye(ff ** 2 * 2).astype(type_complex)

    return kx_vector, ky_vector, Kx, Ky, k_I_z, k_II_z, varphi, Y_I, Y_II, Z_I, Z_II, big_F, big_G, big_T


def transfer_2d_wv(ff, Kx, E_conv_i, Ky, o_E_conv_i, E_conv, type_complex=jnp.complex128, perturbation=1E-10):

    I = ee.eye(ff ** 2).astype(type_complex)

    B = Kx @ E_conv_i @ Kx - I
    D = Ky @ E_conv_i @ Ky - I

    S2_from_S = ee.block(
        [
            [Ky ** 2 + B @ o_E_conv_i, Kx @ (E_conv_i @ Ky @ E_conv - Ky)],
            [Ky @ (E_conv_i @ Kx @ o_E_conv_i - Kx), Kx ** 2 + D @ E_conv]
        ])

    eigenvalues, W = eig(S2_from_S, type_complex=type_complex, perturbation=perturbation)

    q = eigenvalues ** 0.5

    Q = ee.diag(q)
    Q_i = ee.inv(Q)
    U1_from_S = ee.block(
        [
            [-Kx @ Ky, Kx ** 2 - E_conv],
            [o_E_conv_i - Ky ** 2, Ky @ Kx]
        ]
    )
    V = U1_from_S @ W @ Q_i

    return W, V, q


def transfer_2d_2(k0, d, W, V, center, q, varphi, I, O, big_F, big_G, big_T, type_complex=jnp.complex128):

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

    X_1 = ee.diag(ee.exp(-k0 * q1 * d))
    X_2 = ee.diag(ee.exp(-k0 * q2 * d))

    F_c = ee.diag(ee.cos(varphi))
    F_s = ee.diag(ee.sin(varphi))

    W_ss = F_c @ W_21 - F_s @ W_11
    W_sp = F_c @ W_22 - F_s @ W_12
    W_ps = F_c @ W_11 + F_s @ W_21
    W_pp = F_c @ W_12 + F_s @ W_22

    V_ss = F_c @ V_11 + F_s @ V_21
    V_sp = F_c @ V_12 + F_s @ V_22
    V_ps = F_c @ V_21 - F_s @ V_11
    V_pp = F_c @ V_22 - F_s @ V_12

    big_I = ee.eye(2 * (len(I))).astype(type_complex)
    big_X = ee.block([[X_1, O], [O, X_2]])
    big_W = ee.block([[W_ss, W_sp], [W_ps, W_pp]])
    big_V = ee.block([[V_ss, V_sp], [V_ps, V_pp]])

    big_W_i = ee.inv(big_W)
    big_V_i = ee.inv(big_V)

    big_A = 0.5 * (big_W_i @ big_F + big_V_i @ big_G)
    big_B = 0.5 * (big_W_i @ big_F - big_V_i @ big_G)

    big_A_i = ee.inv(big_A)

    big_F = big_W @ (big_I + big_X @ big_B @ big_A_i @ big_X)
    big_G = big_V @ (big_I - big_X @ big_B @ big_A_i @ big_X)

    big_T = big_T @ big_A_i @ big_X

    return big_X, big_F, big_G, big_T, big_A_i, big_B, W_11, W_12, W_21, W_22, V_11, V_12, V_21, V_22


def transfer_2d_3(center, big_F, big_G, big_T, Z_I, Y_I, psi, theta, ff, delta_i0, k_I_z, k0, n_I, n_II, k_II_z,
                  type_complex=jnp.complex128):

    I = ee.eye(ff ** 2).astype(type_complex)
    O = ee.zeros((ff ** 2, ff ** 2), dtype=type_complex)

    big_F_11 = big_F[:center, :center]
    big_F_12 = big_F[:center, center:]
    big_F_21 = big_F[center:, :center]
    big_F_22 = big_F[center:, center:]

    big_G_11 = big_G[:center, :center]
    big_G_12 = big_G[:center, center:]
    big_G_21 = big_G[center:, :center]
    big_G_22 = big_G[center:, center:]

    # Final Equation in form of AX=B
    final_A = ee.block(
        [
            [I, O, -big_F_11, -big_F_12],
            [O, -1j * Z_I, -big_F_21, -big_F_22],
            [-1j * Y_I, O, -big_G_11, -big_G_12],
            [O, I, -big_G_21, -big_G_22],
        ]
    )

    final_B = ee.block(
        [
            [-ee.sin(psi) * delta_i0],
            [-ee.cos(psi) * ee.cos(theta) * delta_i0],
            [-1j * ee.sin(psi) * n_I * ee.cos(theta) * delta_i0],
            [1j * n_I * ee.cos(psi) * delta_i0]
        ]
    )

    final_RT = ee.inv(final_A) @ final_B

    R_s = final_RT[:ff ** 2, :].flatten()
    R_p = final_RT[ff ** 2:2 * ff ** 2, :].flatten()

    big_T1 = final_RT[2 * ff ** 2:, :]
    big_T = big_T @ big_T1

    T_s = big_T[:ff ** 2, :].flatten()
    T_p = big_T[ff ** 2:, :].flatten()

    de_ri = R_s * ee.conj(R_s) * ee.real(k_I_z / (k0 * n_I * ee.cos(theta))) \
            + R_p * ee.conj(R_p) * ee.real((k_I_z / n_I ** 2) / (k0 * n_I * ee.cos(theta)))

    de_ti = T_s * ee.conj(T_s) * ee.real(k_II_z / (k0 * n_I * ee.cos(theta))) \
            + T_p * ee.conj(T_p) * ee.real((k_II_z / n_II ** 2) / (k0 * n_I * ee.cos(theta)))

    return de_ri.real, de_ti.real, big_T1
