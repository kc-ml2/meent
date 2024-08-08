import jax
import jax.numpy as jnp

from .primitives import eig, conj


def transfer_1d_1(pol, ff_x, kx, n_top, n_bot, type_complex=jnp.complex128):

    ff_xy = ff_x * 1

    kz_top = (n_top ** 2 - kx ** 2) ** 0.5
    kz_bot = (n_bot ** 2 - kx ** 2) ** 0.5

    kz_top = kz_top.conjugate()
    kz_bot = kz_bot.conjugate()

    F = jnp.eye(ff_xy, dtype=type_complex)

    def false_fun(kz_bot):
        Kz_bot = jnp.diag(kz_bot)
        G = 1j * Kz_bot
        return Kz_bot, G

    def true_fun(kz_bot):
        Kz_bot = jnp.diag(kz_bot / (n_bot ** 2))
        G = 1j * Kz_bot
        return Kz_bot, G

    Kz_bot, G = jax.lax.cond(pol, true_fun, false_fun, kz_bot)

    T = jnp.eye(ff_xy, dtype=type_complex)

    return kz_top, kz_bot, F, G, T

    # if pol == 0:  # TE
    #     Kz_bot = jnp.diag(kz_bot)
    #
    #     G = 1j * Kz_bot
    #
    # elif pol == 1:  # TM
    #     Kz_bot = jnp.diag(kz_bot / (n_bot ** 2))
    #
    #     G = 1j * Kz_bot
    #
    # else:
    #     raise ValueError
    #
    # T = jnp.eye(ff_xy, dtype=type_complex)
    #
    # return kz_top, kz_bot, F, G, T


def transfer_1d_2(pol, kx, epx_conv, epy_conv, epz_conv_i, type_complex=jnp.complex128):

    Kx = jnp.diag(kx)

    def false_fun(Kx, epy_conv):  # TE
        A = Kx ** 2 - epy_conv
        eigenvalues, W = eig(A)
        eigenvalues += 0j  # to get positive square root
        q = eigenvalues ** 0.5
        Q = jnp.diag(q)
        V = W @ Q
        return W, V, q

    def true_fun(Kx, epy_conv):  # TM
        B = Kx @ epz_conv_i @ Kx - jnp.eye(epy_conv.shape[0], dtype=type_complex)

        eigenvalues, W = eig(epx_conv @ B)

        eigenvalues += 0j  # to get positive square root
        q = eigenvalues ** 0.5

        Q = jnp.diag(q)
        V = jnp.linalg.inv(epx_conv) @ W @ Q
        return W, V, q

    W, V, q = jax.lax.cond(pol, true_fun, false_fun, Kx, epy_conv)

    return W, V, q
    # if pol == 0:
    #     A = Kx ** 2 - epy_conv
    #     eigenvalues, W = eig(A)
    #     eigenvalues += 0j  # to get positive square root
    #     q = eigenvalues ** 0.5
    #     Q = jnp.diag(q)
    #     V = W @ Q
    #
    # elif pol == 1:
    #     B = Kx @ epz_conv_i @ Kx - jnp.eye(epy_conv.shape[0], dtype=type_complex)
    #
    #     eigenvalues, W = eig(epx_conv @ B)
    #
    #     eigenvalues += 0j  # to get positive square root
    #     q = eigenvalues ** 0.5
    #
    #     Q = jnp.diag(q)
    #     V = jnp.linalg.inv(epx_conv) @ W @ Q
    #
    # else:
    #     raise ValueError
    #
    # return W, V, q


def transfer_1d_3(k0, W, V, q, d, F, G, T, type_complex=jnp.complex128):

    ff_x = len(q)

    I = jnp.eye(ff_x, dtype=type_complex)

    X = jnp.diag(jnp.exp(-k0 * q * d))

    W_i = jnp.linalg.inv(W)
    V_i = jnp.linalg.inv(V)

    A = 0.5 * (W_i @ F + V_i @ G)
    B = 0.5 * (W_i @ F - V_i @ G)

    A_i = jnp.linalg.inv(A)

    F = W @ (I + X @ B @ A_i @ X)
    G = V @ (I - X @ B @ A_i @ X)
    T = T @ A_i @ X

    return X, F, G, T, A_i, B


def transfer_1d_4(pol, F, G, T, kz_top, kz_bot, theta, n_top, n_bot, type_complex=jnp.complex128):

    ff_xy = len(kz_top)

    Kz_top = jnp.diag(kz_top)

    delta_i0 = jnp.zeros(ff_xy, dtype=type_complex)
    delta_i0 = delta_i0.at[ff_xy // 2].set(1)

    # if pol == 0:  # TE
    #     inc_term = 1j * n_top * jnp.cos(theta) * delta_i0
    #     T1 = jnp.linalg.inv(G + 1j * Kz_top @ F) @ (1j * Kz_top @ delta_i0 + inc_term)
    #
    # elif pol == 1:  # TM
    #     inc_term = 1j * delta_i0 * jnp.cos(theta) / n_top
    #     T1 = jnp.linalg.inv(G + 1j * Kz_top/(n_top ** 2) @ F) @ (1j * Kz_top/(n_top ** 2) @ delta_i0 + inc_term)

    def false_fun(n_top, theta, delta_i0, G, Kz_top, T):  # TE
        inc_term = 1j * n_top * jnp.cos(theta) * delta_i0
        T1 = jnp.linalg.inv(G + 1j * Kz_top @ F) @ (1j * Kz_top @ delta_i0 + inc_term)
        R = F @ T1 - delta_i0
        T = T @ T1

        de_ri = jnp.real(R * jnp.conj(R) * kz_top / (n_top * jnp.cos(theta)))
        de_ti = T * jnp.conj(T) * jnp.real(kz_bot / (n_top * jnp.cos(theta)))

        return de_ri, de_ti, T1

    def true_fun(n_top, theta, delta_i0, G, Kz_top, T):  # TM
        inc_term = 1j * delta_i0 * jnp.cos(theta) / n_top
        T1 = jnp.linalg.inv(G + 1j * Kz_top / (n_top ** 2) @ F) @ (1j * Kz_top / (n_top ** 2) @ delta_i0 + inc_term)

        R = F @ T1 - delta_i0
        T = T @ T1

        de_ri = jnp.real(R * jnp.conj(R) * kz_top / (n_top * jnp.cos(theta)))
        de_ti = T * jnp.conj(T) * jnp.real(kz_bot / n_bot ** 2) / (jnp.cos(theta) / n_top)

        return de_ri, de_ti, T1

    de_ri, de_ti, T1 = jax.lax.cond(pol, true_fun, false_fun, n_top, theta, delta_i0, G, Kz_top, T)

    # R = F @ T1 - delta_i0
    # T = T @ T1
    #
    # de_ri = jnp.real(R * jnp.conj(R) * kz_top / (n_top * jnp.cos(theta)))
    #
    # if pol == 0:
    #     de_ti = T * jnp.conj(T) * jnp.real(kz_bot / (n_top * jnp.cos(theta)))
    # elif pol == 1:
    #     de_ti = T * jnp.conj(T) * jnp.real(kz_bot / n_bot ** 2) / (jnp.cos(theta) / n_top)
    # else:
    #     raise ValueError

    return de_ri.real, de_ti.real, T1


def transfer_2d_1(ff_x, ff_y, kx, ky, n_top, n_bot, type_complex=jnp.complex128):

    ff_xy = ff_x * ff_y

    I = jnp.eye(ff_xy, dtype=type_complex)
    O = jnp.zeros((ff_xy, ff_xy), dtype=type_complex)

    kz_top = (n_top ** 2 - kx ** 2 - ky.reshape((-1, 1)) ** 2) ** 0.5
    kz_bot = (n_bot ** 2 - kx ** 2 - ky.reshape((-1, 1)) ** 2) ** 0.5

    kz_top = kz_top.flatten().conjugate()
    kz_bot = kz_bot.flatten().conjugate()

    varphi = jnp.arctan(ky.reshape((-1, 1)) / kx).flatten()

    Kz_bot = jnp.diag(kz_bot)

    big_F = jnp.block([[I, O], [O, 1j * Kz_bot / (n_bot ** 2)]])
    big_G = jnp.block([[1j * Kz_bot, O], [O, I]])
    big_T = jnp.eye(2 * ff_xy, dtype=type_complex)

    return kz_top, kz_bot, varphi, big_F, big_G, big_T


def transfer_2d_2(kx, ky, epx_conv, epy_conv, epz_conv_i, type_complex=jnp.complex128):

    ff_x = len(kx)
    ff_y = len(ky)

    I = jnp.eye(ff_y * ff_x, dtype=type_complex)

    Kx = jnp.diag(jnp.tile(kx, ff_y).flatten())
    Ky = jnp.diag(jnp.tile(ky.reshape((-1, 1)), ff_x).flatten())

    B = Kx @ epz_conv_i @ Kx - I
    D = Ky @ epz_conv_i @ Ky - I

    Omega2_LR = jnp.block(
        [
            [Ky ** 2 + B @ epx_conv, Kx @ (epz_conv_i @ Ky @ epy_conv - Ky)],
            [Ky @ (epz_conv_i @ Kx @ epx_conv - Kx), Kx ** 2 + D @ epy_conv]
        ])

    # eigenvalues, W = jnp.linalg.eig(Omega2_LR)
    eigenvalues, W = eig(Omega2_LR)
    eigenvalues += 0j  # to get positive square root
    q = eigenvalues ** 0.5

    Q = jnp.diag(q)
    Q_i = jnp.linalg.inv(Q)

    Omega_R = jnp.block(
        [
            [-Kx @ Ky, Kx ** 2 - epy_conv],
            [epx_conv - Ky ** 2, Ky @ Kx]
        ]
    )

    V = Omega_R @ W @ Q_i

    return W, V, q


def transfer_2d_3(k0, W, V, q, d, varphi, big_F, big_G, big_T, type_complex=jnp.complex128):

    ff_xy = len(q)//2

    I = jnp.eye(ff_xy, dtype=type_complex)
    O = jnp.zeros((ff_xy, ff_xy), dtype=type_complex)

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

    X_1 = jnp.diag(jnp.exp(-k0 * q1 * d))
    X_2 = jnp.diag(jnp.exp(-k0 * q2 * d))

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

    big_I = jnp.eye(2 * (len(I)), dtype=type_complex)
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

    return big_X, big_F, big_G, big_T, big_A_i, big_B


def transfer_2d_4(big_F, big_G, big_T, kz_top, kz_bot, psi, theta, n_top, n_bot, type_complex=jnp.complex128):

    ff_xy = len(big_F) // 2

    Kz_top = jnp.diag(kz_top)

    I = jnp.eye(ff_xy, dtype=type_complex)
    O = jnp.zeros((ff_xy, ff_xy), dtype=type_complex)

    big_F_11 = big_F[:ff_xy, :ff_xy]
    big_F_12 = big_F[:ff_xy, ff_xy:]
    big_F_21 = big_F[ff_xy:, :ff_xy]
    big_F_22 = big_F[ff_xy:, ff_xy:]

    big_G_11 = big_G[:ff_xy, :ff_xy]
    big_G_12 = big_G[:ff_xy, ff_xy:]
    big_G_21 = big_G[ff_xy:, :ff_xy]
    big_G_22 = big_G[ff_xy:, ff_xy:]

    delta_i0 = jnp.zeros((ff_xy, 1), dtype=type_complex)
    delta_i0 = delta_i0.at[ff_xy // 2, 0].set(1)

    # Final Equation in form of AX=B
    final_A = jnp.block(
        [
            [I, O, -big_F_11, -big_F_12],
            [O, -1j * Kz_top / (n_top ** 2), -big_F_21, -big_F_22],
            [-1j * Kz_top, O, -big_G_11, -big_G_12],
            [O, I, -big_G_21, -big_G_22],
        ]
    )

    final_B = jnp.block(
        [
            [-jnp.sin(psi) * delta_i0],
            [jnp.cos(psi) * jnp.cos(theta) * delta_i0],
            [-1j * jnp.sin(psi) * n_top * jnp.cos(theta) * delta_i0],
            [-1j * n_top * jnp.cos(psi) * delta_i0]
        ]
    )

    final_RT = jnp.linalg.inv(final_A) @ final_B

    R_s = final_RT[:ff_xy, :].flatten()
    R_p = final_RT[ff_xy: 2 * ff_xy, :].flatten()

    big_T1 = final_RT[2 * ff_xy:, :]
    big_T = big_T @ big_T1

    T_s = big_T[:ff_xy, :].flatten()
    T_p = big_T[ff_xy:, :].flatten()

    de_ri = R_s * jnp.conj(R_s) * jnp.real(kz_top / (n_top * jnp.cos(theta))) \
            + R_p * jnp.conj(R_p) * jnp.real(kz_top / n_top ** 2 / (n_top * jnp.cos(theta)))

    de_ti = T_s * jnp.conj(T_s) * jnp.real(kz_bot / (n_top * jnp.cos(theta))) \
            + T_p * jnp.conj(T_p) * jnp.real(kz_bot / n_bot ** 2 / (n_top * jnp.cos(theta)))

    return de_ri.real, de_ti.real, big_T1

