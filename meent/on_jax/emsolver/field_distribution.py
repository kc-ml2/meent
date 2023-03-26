import jax
import jax.numpy as jnp

from functools import partial


def field_distribution(grating_type, *args, **kwargs):
    if grating_type == 0:
        res = field_dist_1d_vectorized_ji(*args, **kwargs)
    elif grating_type == 1:
        res = field_dist_1d_conical_vectorized_ji(*args, **kwargs)
    else:
        res = field_dist_2d_vectorized_ji(*args, **kwargs)
    return res


@partial(jax.jit, static_argnums=(5, 6, 7))
def field_dist_1d_vectorized_ji(wavelength, kx_vector, T1, layer_info_list, period,
                                pol, resolution=(100, 1, 100), type_complex=jnp.complex128):

    k0 = 2 * jnp.pi / wavelength
    Kx = jnp.diag(kx_vector / k0)

    resolution_x, resolution_y, resolution_z = resolution
    field_cell = jnp.zeros((resolution_z * len(layer_info_list), resolution_y, resolution_x, 3), dtype=type_complex)

    T_layer = T1

    # From the first layer
    for idx_layer, (E_conv_i, q, W, X, a_i, b, d) in enumerate(layer_info_list[::-1]):

        c1 = T_layer[:, None]
        c2 = b @ a_i @ X @ T_layer[:, None]

        Q = jnp.diag(q)

        if pol == 0:
            V = W @ Q
            EKx = None

        else:
            V = E_conv_i @ W @ Q
            EKx = E_conv_i @ Kx

        for k in range(resolution_z):
            z = k / resolution_z * d

            if pol == 0:
                Sy = W @ (diag_exp(-k0 * Q * z) @ c1 + diag_exp(k0 * Q * (z - d)) @ c2)
                Ux = V @ (-diag_exp(-k0 * Q * z) @ c1 + diag_exp(k0 * Q * (z - d)) @ c2)
                C = Kx @ Sy

                x_1d = jnp.arange(resolution_x).reshape((1, -1, 1))
                x_1d = -1j * x_1d * period[0] / resolution_x
                x_2d = jnp.tile(x_1d, (resolution_y, 1, 1))
                x_2d = x_2d * kx_vector
                x_2d = x_2d.reshape((resolution_y, resolution_x, 1, len(kx_vector)))

                exp_K = jnp.exp(x_2d)
                exp_K = exp_K.reshape((resolution_y, resolution_x, -1))

                Ey = exp_K @ Sy
                Hx = -1j * exp_K @ Ux
                Hz = -1j * exp_K @ C

                val = jnp.concatenate((Ey, Hx, Hz), axis=-1)

            else:
                Uy = W @ (diag_exp(-k0 * Q * z) @ c1 + diag_exp(k0 * Q * (z - d)) @ c2)
                Sx = V @ (-diag_exp(-k0 * Q * z) @ c1 + diag_exp(k0 * Q * (z - d)) @ c2)

                C = EKx @ Uy  # there is a better option for convergence

                x_1d = jnp.arange(resolution_x).reshape((1, -1, 1))
                x_1d = -1j * x_1d * period[0] / resolution_x
                x_2d = jnp.tile(x_1d, (resolution_y, 1, 1))
                x_2d = x_2d * kx_vector
                x_2d = x_2d.reshape((resolution_y, resolution_x, 1, len(kx_vector)))

                exp_K = jnp.exp(x_2d)
                exp_K = exp_K.reshape((resolution_y, resolution_x, -1))

                Hy = exp_K @ Uy
                Ex = 1j * exp_K @ Sx
                Ez = -1j * exp_K @ C

                val = jnp.concatenate((Hy, Ex, Ez), axis=-1)

            field_cell = field_cell.at[resolution_z * idx_layer + k].set(val)

        T_layer = a_i @ X @ T_layer

    return field_cell


@partial(jax.jit, static_argnums=(8, 9))
def field_dist_1d_conical_vectorized_ji(wavelength, kx_vector, n_I, theta, phi, T1, layer_info_list, period,
                                        resolution=(100, 100, 100), type_complex=jnp.complex128):

    k0 = 2 * jnp.pi / wavelength
    ky = k0 * n_I * jnp.sin(theta) * jnp.sin(phi)
    Kx = jnp.diag(kx_vector / k0)

    resolution_x, resolution_y, resolution_z = resolution
    field_cell = jnp.zeros((resolution_z * len(layer_info_list), resolution_y, resolution_x, 6), dtype=type_complex)

    T_layer = T1

    big_I = jnp.eye((len(T1))).astype(type_complex)

    # From the first layer
    for idx_layer, [E_conv_i, q_1, q_2, W_1, W_2, V_11, V_12, V_21, V_22, big_X, big_A_i, big_B, d] \
            in enumerate(layer_info_list[::-1]):

        c = jnp.block([[big_I], [big_B @ big_A_i @ big_X]]) @ T_layer

        ff = len(c) // 4

        c1_plus = c[0 * ff:1 * ff]
        c2_plus = c[1 * ff:2 * ff]
        c1_minus = c[2 * ff:3 * ff]
        c2_minus = c[3 * ff:4 * ff]

        big_Q1 = jnp.diag(q_1)
        big_Q2 = jnp.diag(q_2)
        for k in range(resolution_z):
            # Sx, Sy, Ux, Uy, Sz, Uz = z_loop_1d_conical(k, c, k0, Kx, ky, resolution_z, E_conv_i, q_1, q_2, W_1, W_2, V_11, V_12, V_21, V_22, d)

            z = k / resolution_z * d

            Sx = W_2 @ (diag_exp(-k0 * big_Q2 * z) @ c2_plus + diag_exp(k0 * big_Q2 * (z - d)) @ c2_minus)

            Sy = V_11 @ (diag_exp(-k0 * big_Q1 * z) @ c1_plus + diag_exp(k0 * big_Q1 * (z - d)) @ c1_minus) \
                 + V_12 @ (diag_exp(-k0 * big_Q2 * z) @ c2_plus + diag_exp(k0 * big_Q2 * (z - d)) @ c2_minus)

            Ux = W_1 @ (-diag_exp(-k0 * big_Q1 * z) @ c1_plus + diag_exp(k0 * big_Q1 * (z - d)) @ c1_minus)

            Uy = V_21 @ (-diag_exp(-k0 * big_Q1 * z) @ c1_plus + diag_exp(k0 * big_Q1 * (z - d)) @ c1_minus) \
                 + V_22 @ (-diag_exp(-k0 * big_Q2 * z) @ c2_plus + diag_exp(k0 * big_Q2 * (z - d)) @ c2_minus)

            Sz = -1j * E_conv_i @ (Kx @ Uy - ky * Ux)

            Uz = -1j * (Kx @ Sy - ky * Sx)

            x_1d = jnp.arange(resolution_x).reshape((1, -1, 1))
            x_1d = -1j * x_1d * period[0] / resolution_x
            x_2d = jnp.tile(x_1d, (resolution_y, 1, 1))
            x_2d = x_2d * kx_vector
            x_2d = x_2d.reshape((resolution_y, resolution_x, 1, len(kx_vector)))

            exp_K = jnp.exp(x_2d)
            exp_K = exp_K.reshape((resolution_y, resolution_x, -1))

            Ex = exp_K @ Sx
            Ey = exp_K @ Sy
            Ez = exp_K @ Sz

            Hx = -1j * exp_K @ Ux
            Hy = -1j * exp_K @ Uy
            Hz = -1j * exp_K @ Uz

            val = jnp.concatenate((Ex, Ey, Ez, Hx, Hy, Hz), axis=-1)

            field_cell = field_cell.at[resolution_z * idx_layer + k].set(val)

        T_layer = big_A_i @ big_X @ T_layer

    return field_cell


@partial(jax.jit, static_argnums=(5, 6, 10, 11))
def field_dist_2d_vectorized_ji(wavelength, kx_vector, n_I, theta, phi, fourier_order_x, fourier_order_y, T1, layer_info_list, period,
                                resolution=(10, 10, 10), type_complex=jnp.complex128):

    k0 = 2 * jnp.pi / wavelength

    fourier_indices_y = jnp.arange(-fourier_order_y, fourier_order_y + 1)
    ff_x = fourier_order_x * 2 + 1
    ff_y = fourier_order_y * 2 + 1
    ky_vector = k0 * (n_I * jnp.sin(theta) * jnp.sin(phi) + fourier_indices_y * (
            wavelength / period[1])).astype(type_complex)

    Kx = jnp.diag(jnp.tile(kx_vector, ff_y).flatten()) / k0
    Ky = jnp.diag(jnp.tile(ky_vector.reshape((-1, 1)), ff_x).flatten()) / k0

    resolution_x, resolution_y, resolution_z = resolution
    field_cell = jnp.zeros((resolution_z * len(layer_info_list), resolution_y, resolution_x, 6), dtype=type_complex)

    T_layer = T1

    big_I = jnp.eye((len(T1))).astype(type_complex)

    # From the first layer
    for idx_layer, (E_conv_i, q, W_11, W_12, W_21, W_22, V_11, V_12, V_21, V_22, big_X, big_A_i, big_B, d)\
            in enumerate(layer_info_list[::-1]):

        c = jnp.block([[big_I], [big_B @ big_A_i @ big_X]]) @ T_layer

        ff = len(c) // 4

        c1_plus = c[0 * ff:1 * ff]
        c2_plus = c[1 * ff:2 * ff]
        c1_minus = c[2 * ff:3 * ff]
        c2_minus = c[3 * ff:4 * ff]

        q1 = q[:len(q) // 2]
        q2 = q[len(q) // 2:]
        big_Q1 = jnp.diag(q1)
        big_Q2 = jnp.diag(q2)
        for k in range(resolution_z):
            z = k / resolution_z * d
            Sx = W_11 @ (diag_exp(-k0 * big_Q1 * z) @ c1_plus + diag_exp(k0 * big_Q1 * (z - d)) @ c1_minus) \
                 + W_12 @ (diag_exp(-k0 * big_Q2 * z) @ c2_plus + diag_exp(k0 * big_Q2 * (z - d)) @ c2_minus)

            Sy = W_21 @ (diag_exp(-k0 * big_Q1 * z) @ c1_plus + diag_exp(k0 * big_Q1 * (z - d)) @ c1_minus) \
                 + W_22 @ (diag_exp(-k0 * big_Q2 * z) @ c2_plus + diag_exp(k0 * big_Q2 * (z - d)) @ c2_minus)

            Ux = V_11 @ (-diag_exp(-k0 * big_Q1 * z) @ c1_plus + diag_exp(k0 * big_Q1 * (z - d)) @ c1_minus) \
                 + V_12 @ (-diag_exp(-k0 * big_Q2 * z) @ c2_plus + diag_exp(k0 * big_Q2 * (z - d)) @ c2_minus)

            Uy = V_21 @ (-diag_exp(-k0 * big_Q1 * z) @ c1_plus + diag_exp(k0 * big_Q1 * (z - d)) @ c1_minus) \
                 + V_22 @ (-diag_exp(-k0 * big_Q2 * z) @ c2_plus + diag_exp(k0 * big_Q2 * (z - d)) @ c2_minus)

            Sz = -1j * E_conv_i @ (Kx @ Uy - Ky @ Ux)

            Uz = -1j * (Kx @ Sy - Ky @ Sx)

            x_1d = jnp.arange(resolution_x).reshape((1, -1, 1))
            y_1d = jnp.arange(resolution_y).reshape((-1, 1, 1))

            x_1d = -1j * x_1d * period[0] / resolution_x
            y_1d = -1j * y_1d * period[1] / resolution_y

            x_2d = jnp.tile(x_1d, (resolution_y, 1, 1))
            y_2d = jnp.tile(y_1d, (1, resolution_x, 1))

            x_2d = x_2d * kx_vector
            y_2d = y_2d * ky_vector

            x_2d = x_2d.reshape((resolution_y, resolution_x, 1, len(kx_vector)))
            y_2d = y_2d.reshape((resolution_y, resolution_x, len(ky_vector), 1))

            exp_K = jnp.exp(x_2d) * jnp.exp(y_2d)
            exp_K = exp_K.reshape((resolution_y, resolution_x, -1))

            Ex = exp_K @ Sx
            Ey = exp_K @ Sy
            Ez = exp_K @ Sz

            Hx = -1j * exp_K @ Ux
            Hy = -1j * exp_K @ Uy
            Hz = -1j * exp_K @ Uz

            val = jnp.concatenate((Ex, Ey, Ez, Hx, Hy, Hz), axis=-1)

            field_cell = field_cell.at[resolution_z * idx_layer + k].set(val)

        T_layer = big_A_i @ big_X @ T_layer

    return field_cell


@partial(jax.jit, static_argnums=(5, 6, 7))
def field_dist_1d_vectorized_kji(wavelength, kx_vector, T1, layer_info_list, period,
                                 pol, resolution=(100, 1, 100), type_complex=jnp.complex128):

    k0 = 2 * jnp.pi / wavelength
    Kx = jnp.diag(kx_vector / k0)

    resolution_x, resolution_y, resolution_z = resolution
    field_cell = jnp.zeros((resolution_z * len(layer_info_list), resolution_y, resolution_x, 3), dtype=type_complex)

    T_layer = T1

    # From the first layer
    for idx_layer, (E_conv_i, q, W, X, a_i, b, d) in enumerate(layer_info_list[::-1]):

        c1 = T_layer[:, None]
        c2 = b @ a_i @ X @ T_layer[:, None]

        Q = jnp.diag(q)

        if pol == 0:
            V = W @ Q
            EKx = None

        else:
            V = E_conv_i @ W @ Q
            EKx = E_conv_i @ Kx

        z_1d = jnp.arange(resolution_z).reshape((-1, 1, 1)) / resolution_z * d

        if pol == 0:
            Sy = W @ (diag_exp_batch(-k0 * Q * z_1d) @ c1 + diag_exp_batch(k0 * Q * (z_1d - d)) @ c2)
            Ux = V @ (-diag_exp_batch(-k0 * Q * z_1d) @ c1 + diag_exp_batch(k0 * Q * (z_1d - d)) @ c2)
            C = Kx @ Sy

            x_1d = jnp.arange(resolution_x).reshape((1, -1, 1))
            x_1d = -1j * x_1d * period[0] / resolution_x
            x_2d = jnp.tile(x_1d, (resolution_y, 1, 1))
            x_2d = x_2d * kx_vector
            x_2d = x_2d.reshape((resolution_y, resolution_x, 1, len(kx_vector)))

            exp_K = jnp.exp(x_2d)
            exp_K = exp_K.reshape((resolution_y, resolution_x, -1))

            Ey = exp_K[:, :, None, :] @ Sy[:, None, None, :, :]
            Hx = -1j * exp_K[:, :, None, :] @ Ux[:, None, None, :, :]
            Hz = -1j * exp_K[:, :, None, :] @ C[:, None, None, :, :]

            val = jnp.concatenate((Ey.squeeze(-1), Hx.squeeze(-1), Hz.squeeze(-1)), axis=-1)

        else:
            Uy = W @ (diag_exp_batch(-k0 * Q * z_1d) @ c1 + diag_exp_batch(k0 * Q * (z_1d - d)) @ c2)
            Sx = V @ (-diag_exp_batch(-k0 * Q * z_1d) @ c1 + diag_exp_batch(k0 * Q * (z_1d - d)) @ c2)

            C = EKx @ Uy  # there is a better option for convergence

            x_1d = jnp.arange(resolution_x).reshape((1, -1, 1))
            x_1d = -1j * x_1d * period[0] / resolution_x
            x_2d = jnp.tile(x_1d, (resolution_y, 1, 1))
            x_2d = x_2d * kx_vector
            x_2d = x_2d.reshape((resolution_y, resolution_x, 1, len(kx_vector)))

            exp_K = jnp.exp(x_2d)
            exp_K = exp_K.reshape((resolution_y, resolution_x, -1))

            Hy = exp_K[:, :, None, :] @ Uy[:, None, None, :, :]
            Ex = 1j * exp_K[:, :, None, :] @ Sx[:, None, None, :, :]
            Ez = -1j * exp_K[:, :, None, :] @ C[:, None, None, :, :]

            val = jnp.concatenate((Hy.squeeze(-1), Ex.squeeze(-1), Ez.squeeze(-1)), axis=-1)

        field_cell = field_cell.at[resolution_z * idx_layer:resolution_z * (idx_layer + 1)].set(val)

        T_layer = a_i @ X @ T_layer

    return field_cell


@partial(jax.jit, static_argnums=(8, 9))
def field_dist_1d_conical_vectorized_kji(wavelength, kx_vector, n_I, theta, phi, T1, layer_info_list, period,
                                         resolution=(100, 100, 100), type_complex=jnp.complex128):

    k0 = 2 * jnp.pi / wavelength
    ky = k0 * n_I * jnp.sin(theta) * jnp.sin(phi)
    Kx = jnp.diag(kx_vector / k0)

    resolution_x, resolution_y, resolution_z = resolution
    field_cell = jnp.zeros((resolution_z * len(layer_info_list), resolution_y, resolution_x, 6), dtype=type_complex)

    T_layer = T1

    big_I = jnp.eye((len(T1))).astype(type_complex)

    # From the first layer
    for idx_layer, [E_conv_i, q_1, q_2, W_1, W_2, V_11, V_12, V_21, V_22, big_X, big_A_i, big_B, d] \
            in enumerate(layer_info_list[::-1]):

        c = jnp.block([[big_I], [big_B @ big_A_i @ big_X]]) @ T_layer

        z_1d = jnp.arange(resolution_z).reshape((-1, 1, 1)) / resolution_z * d

        ff = len(c) // 4

        c1_plus = c[0 * ff:1 * ff]
        c2_plus = c[1 * ff:2 * ff]
        c1_minus = c[2 * ff:3 * ff]
        c2_minus = c[3 * ff:4 * ff]

        big_Q1 = jnp.diag(q_1)
        big_Q2 = jnp.diag(q_2)

        Sx = W_2 @ (diag_exp_batch(-k0 * big_Q2 * z_1d) @ c2_plus + diag_exp_batch(k0 * big_Q2 * (z_1d - d)) @ c2_minus)

        Sy = V_11 @ (diag_exp_batch(-k0 * big_Q1 * z_1d) @ c1_plus + diag_exp_batch(k0 * big_Q1 * (z_1d - d)) @ c1_minus) \
             + V_12 @ (diag_exp_batch(-k0 * big_Q2 * z_1d) @ c2_plus + diag_exp_batch(k0 * big_Q2 * (z_1d - d)) @ c2_minus)

        Ux = W_1 @ (-diag_exp_batch(-k0 * big_Q1 * z_1d) @ c1_plus + diag_exp_batch(k0 * big_Q1 * (z_1d - d)) @ c1_minus)

        Uy = V_21 @ (-diag_exp_batch(-k0 * big_Q1 * z_1d) @ c1_plus + diag_exp_batch(k0 * big_Q1 * (z_1d - d)) @ c1_minus) \
             + V_22 @ (-diag_exp_batch(-k0 * big_Q2 * z_1d) @ c2_plus + diag_exp_batch(k0 * big_Q2 * (z_1d - d)) @ c2_minus)

        Sz = -1j * E_conv_i @ (Kx @ Uy - ky * Ux)
        Uz = -1j * (Kx @ Sy - ky * Sx)

        x_1d = jnp.arange(resolution_x).reshape((1, -1, 1))
        x_1d = -1j * x_1d * period[0] / resolution_x
        x_2d = jnp.tile(x_1d, (resolution_y, 1, 1))
        x_2d = x_2d * kx_vector
        x_2d = x_2d.reshape((resolution_y, resolution_x, 1, len(kx_vector)))

        exp_K = jnp.exp(x_2d)
        exp_K = exp_K.reshape((resolution_y, resolution_x, -1))

        Ex = exp_K[:, :, None, :] @ Sx[:, None, None, :, :]
        Ey = exp_K[:, :, None, :] @ Sy[:, None, None, :, :]
        Ez = exp_K[:, :, None, :] @ Sz[:, None, None, :, :]

        Hx = -1j * exp_K[:, :, None, :] @ Ux[:, None, None, :, :]
        Hy = -1j * exp_K[:, :, None, :] @ Uy[:, None, None, :, :]
        Hz = -1j * exp_K[:, :, None, :] @ Uz[:, None, None, :, :]

        val = jnp.concatenate(
            (Ex.squeeze(-1), Ey.squeeze(-1), Ez.squeeze(-1), Hx.squeeze(-1), Hy.squeeze(-1), Hz.squeeze(-1)),
            axis=-1)
        field_cell = field_cell.at[resolution_z * idx_layer:resolution_z * (idx_layer + 1)].set(val)
        T_layer = big_A_i @ big_X @ T_layer

    return field_cell

@partial(jax.jit, static_argnums=(5, 6, 10, 11))
def field_dist_2d_vectorized_kji(wavelength, kx_vector, n_I, theta, phi, fourier_order_x, fourier_order_y, T1, layer_info_list, period,
                                 resolution=(10, 10, 10), type_complex=jnp.complex128):

    k0 = 2 * jnp.pi / wavelength

    fourier_indices_y = jnp.arange(-fourier_order_y, fourier_order_y + 1)
    ff_x = fourier_order_x * 2 + 1
    ff_y = fourier_order_y * 2 + 1
    ky_vector = k0 * (n_I * jnp.sin(theta) * jnp.sin(phi) + fourier_indices_y * (
            wavelength / period[1])).astype(type_complex)

    Kx = jnp.diag(jnp.tile(kx_vector, ff_y).flatten()) / k0
    Ky = jnp.diag(jnp.tile(ky_vector.reshape((-1, 1)), ff_x).flatten()) / k0

    resolution_x, resolution_y, resolution_z = resolution
    field_cell = jnp.zeros((resolution_z * len(layer_info_list), resolution_y, resolution_x, 6), dtype=type_complex)

    T_layer = T1

    big_I = jnp.eye((len(T1))).astype(type_complex)

    # From the first layer
    for idx_layer, (E_conv_i, q, W_11, W_12, W_21, W_22, V_11, V_12, V_21, V_22, big_X, big_A_i, big_B, d)\
            in enumerate(layer_info_list[::-1]):

        c = jnp.block([[big_I], [big_B @ big_A_i @ big_X]]) @ T_layer
        z_1d = jnp.arange(resolution_z).reshape((-1, 1, 1)) / resolution_z * d

        ff = len(c) // 4

        c1_plus = c[0 * ff:1 * ff]
        c2_plus = c[1 * ff:2 * ff]
        c1_minus = c[2 * ff:3 * ff]
        c2_minus = c[3 * ff:4 * ff]

        q1 = q[:len(q) // 2]
        q2 = q[len(q) // 2:]
        big_Q1 = jnp.diag(q1)
        big_Q2 = jnp.diag(q2)

        Sx = W_11 @ (diag_exp_batch(-k0 * big_Q1 * z_1d) @ c1_plus + diag_exp_batch(k0 * big_Q1 * (z_1d - d)) @ c1_minus) \
              + W_12 @ (diag_exp_batch(-k0 * big_Q2 * z_1d) @ c2_plus + diag_exp_batch(k0 * big_Q2 * (z_1d - d)) @ c2_minus)

        Sy = W_21 @ (diag_exp_batch(-k0 * big_Q1 * z_1d) @ c1_plus + diag_exp_batch(k0 * big_Q1 * (z_1d - d)) @ c1_minus) \
              + W_22 @ (diag_exp_batch(-k0 * big_Q2 * z_1d) @ c2_plus + diag_exp_batch(k0 * big_Q2 * (z_1d - d)) @ c2_minus)

        Ux = V_11 @ (-diag_exp_batch(-k0 * big_Q1 * z_1d) @ c1_plus + diag_exp_batch(k0 * big_Q1 * (z_1d - d)) @ c1_minus) \
              + V_12 @ (-diag_exp_batch(-k0 * big_Q2 * z_1d) @ c2_plus + diag_exp_batch(k0 * big_Q2 * (z_1d - d)) @ c2_minus)

        Uy = V_21 @ (-diag_exp_batch(-k0 * big_Q1 * z_1d) @ c1_plus + diag_exp_batch(k0 * big_Q1 * (z_1d - d)) @ c1_minus) \
              + V_22 @ (-diag_exp_batch(-k0 * big_Q2 * z_1d) @ c2_plus + diag_exp_batch(k0 * big_Q2 * (z_1d - d)) @ c2_minus)

        Sz = -1j * E_conv_i @ (Kx @ Uy - Ky @ Ux)
        Uz = -1j * (Kx @ Sy - Ky @ Sx)

        x_1d = jnp.arange(resolution_x).reshape((1, -1, 1))
        y_1d = jnp.arange(resolution_y).reshape((-1, 1, 1))

        x_1d = -1j * x_1d * period[0] / resolution_x
        y_1d = -1j * y_1d * period[1] / resolution_y

        x_2d = jnp.tile(x_1d, (resolution_y, 1, 1))
        y_2d = jnp.tile(y_1d, (1, resolution_x, 1))

        x_2d = x_2d * kx_vector
        y_2d = y_2d * ky_vector

        x_2d = x_2d.reshape((resolution_y, resolution_x, 1, len(kx_vector)))
        y_2d = y_2d.reshape((resolution_y, resolution_x, len(ky_vector), 1))

        exp_K = jnp.exp(x_2d) * jnp.exp(y_2d)
        exp_K = exp_K.reshape((resolution_y, resolution_x, -1))

        Ex = exp_K[:, :, None, :] @ Sx[:, None, None, :, :]
        Ey = exp_K[:, :, None, :] @ Sy[:, None, None, :, :]
        Ez = exp_K[:, :, None, :] @ Sz[:, None, None, :, :]

        Hx = -1j * exp_K[:, :, None, :] @ Ux[:, None, None, :, :]
        Hy = -1j * exp_K[:, :, None, :] @ Uy[:, None, None, :, :]
        Hz = -1j * exp_K[:, :, None, :] @ Uz[:, None, None, :, :]

        val = jnp.concatenate(
            (Ex.squeeze(-1), Ey.squeeze(-1), Ez.squeeze(-1), Hx.squeeze(-1), Hy.squeeze(-1), Hz.squeeze(-1)),
            axis=-1)

        field_cell = field_cell.at[resolution_z * idx_layer:resolution_z * (idx_layer + 1)].set(val)

        T_layer = big_A_i @ big_X @ T_layer

    return field_cell


def field_dist_1d_vanilla(wavelength, kx_vector, T1, layer_info_list, period,
                          pol, resolution=(100, 1, 100), type_complex=jnp.complex128):

    k0 = 2 * jnp.pi / wavelength
    Kx = jnp.diag(kx_vector / k0)

    resolution_x, resolution_y, resolution_z = resolution
    field_cell = jnp.zeros((resolution_z * len(layer_info_list), resolution_y, resolution_x, 3), dtype=type_complex)

    T_layer = T1

    # From the first layer
    for idx_layer, (E_conv_i, q, W, X, a_i, b, d) in enumerate(layer_info_list[::-1]):

        c1 = T_layer[:, None]
        c2 = b @ a_i @ X @ T_layer[:, None]

        Q = jnp.diag(q)

        if pol == 0:
            V = W @ Q
            EKx = None

        else:
            V = E_conv_i @ W @ Q
            EKx = E_conv_i @ Kx

        for k in range(resolution_z):
            z = k / resolution_z * d

            if pol == 0:
                Sy = W @ (diag_exp(-k0 * Q * z) @ c1 + diag_exp(k0 * Q * (z - d)) @ c2)
                Ux = V @ (-diag_exp(-k0 * Q * z) @ c1 + diag_exp(k0 * Q * (z - d)) @ c2)
                C = Kx @ Sy

                for j in range(resolution_y):
                    for i in range(resolution_x):
                        x = i * period[0] / resolution_x

                        Ey = Sy.T @ jnp.exp(-1j * kx_vector.reshape((-1, 1)) * x)
                        Hx = -1j * Ux.T @ jnp.exp(-1j * kx_vector.reshape((-1, 1)) * x)
                        Hz = -1j * C.T @ jnp.exp(-1j * kx_vector.reshape((-1, 1)) * x)

                        field_cell = field_cell.at[resolution_z * idx_layer + k, j, i].set([Ey[0, 0], Hx[0, 0], Hz[0, 0]])

            else:
                Uy = W @ (diag_exp(-k0 * Q * z) @ c1 + diag_exp(k0 * Q * (z - d)) @ c2)
                Sx = V @ (-diag_exp(-k0 * Q * z) @ c1 + diag_exp(k0 * Q * (z - d)) @ c2)

                C = EKx @ Uy  # there is a better option for convergence
                for j in range(resolution_y):
                    for i in range(resolution_x):
                        x = i * period[0] / resolution_x

                        Hy = Uy.T @ jnp.exp(-1j * kx_vector.reshape((-1, 1)) * x)
                        Ex = 1j * Sx.T @ jnp.exp(-1j * kx_vector.reshape((-1, 1)) * x)
                        Ez = (-1j) * C.T @ jnp.exp(-1j * kx_vector.reshape((-1, 1)) * x)

                        field_cell = field_cell.at[resolution_z * idx_layer + k, j, i].set([Hy[0, 0], Ex[0, 0], Ez[0, 0]])

        T_layer = a_i @ X @ T_layer

    return field_cell


def field_dist_1d_conical_vanilla(wavelength, kx_vector, n_I, theta, phi, T1, layer_info_list, period,
                                  resolution=(100, 100, 100), type_complex=jnp.complex128):

    k0 = 2 * jnp.pi / wavelength
    ky = k0 * n_I * jnp.sin(theta) * jnp.sin(phi)
    Kx = jnp.diag(kx_vector / k0)

    resolution_x, resolution_y, resolution_z = resolution
    field_cell = jnp.zeros((resolution_z * len(layer_info_list), resolution_y, resolution_x, 6), dtype=type_complex)

    T_layer = T1

    big_I = jnp.eye((len(T1))).astype(type_complex)

    # From the first layer
    for idx_layer, [E_conv_i, q_1, q_2, W_1, W_2, V_11, V_12, V_21, V_22, big_X, big_A_i, big_B, d] \
            in enumerate(layer_info_list[::-1]):

        c = jnp.block([[big_I], [big_B @ big_A_i @ big_X]]) @ T_layer

        ff = len(c) // 4

        c1_plus = c[0 * ff:1 * ff]
        c2_plus = c[1 * ff:2 * ff]
        c1_minus = c[2 * ff:3 * ff]
        c2_minus = c[3 * ff:4 * ff]

        big_Q1 = jnp.diag(q_1)
        big_Q2 = jnp.diag(q_2)

        for k in range(resolution_z):
            z = k / resolution_z * d

            Sx = W_2 @ (diag_exp(-k0 * big_Q2 * z) @ c2_plus + diag_exp(k0 * big_Q2 * (z - d)) @ c2_minus)
            Sy = V_11 @ (diag_exp(-k0 * big_Q1 * z) @ c1_plus + diag_exp(k0 * big_Q1 * (z - d)) @ c1_minus) \
                 + V_12 @ (diag_exp(-k0 * big_Q2 * z) @ c2_plus + diag_exp(k0 * big_Q2 * (z - d)) @ c2_minus)
            Ux = W_1 @ (-diag_exp(-k0 * big_Q1 * z) @ c1_plus + diag_exp(k0 * big_Q1 * (z - d)) @ c1_minus)
            Uy = V_21 @ (-diag_exp(-k0 * big_Q1 * z) @ c1_plus + diag_exp(k0 * big_Q1 * (z - d)) @ c1_minus) \
                 + V_22 @ (-diag_exp(-k0 * big_Q2 * z) @ c2_plus + diag_exp(k0 * big_Q2 * (z - d)) @ c2_minus)
            Sz = -1j * E_conv_i @ (Kx @ Uy - ky * Ux)
            Uz = -1j * (Kx @ Sy - ky * Sx)

            for j in range(resolution_y):
                for i in range(resolution_x):
                    # val = x_loop_1d_conical(period, resolution_x, kx_vector, Sx, Sy, Sz, Ux, Uy, Uz, i)
                    x = i * period[0] / resolution_x

                    exp_K = jnp.exp(-1j * kx_vector.reshape((-1, 1)) * x)
                    # exp_K = exp_K.flatten()

                    Ex = Sx.T @ exp_K
                    Ey = Sy.T @ exp_K
                    Ez = Sz.T @ exp_K

                    Hx = -1j * Ux.T @ exp_K
                    Hy = -1j * Uy.T @ exp_K
                    Hz = -1j * Uz.T @ exp_K

                    val = [Ex[0, 0], Ey[0, 0], Ez[0, 0], Hx[0, 0], Hy[0, 0], Hz[0, 0]]
                    field_cell = field_cell.at[resolution_z * idx_layer + k, j, i].set(val)

        T_layer = big_A_i @ big_X @ T_layer

    return field_cell


def field_dist_2d_vanilla(wavelength, kx_vector, n_I, theta, phi, fourier_order_x, fourier_order_y, T1, layer_info_list, period,
                          resolution, type_complex=jnp.complex128):

    k0 = 2 * jnp.pi / wavelength

    fourier_indices_y = jnp.arange(-fourier_order_y, fourier_order_y + 1)
    ff_x = fourier_order_x * 2 + 1
    ff_y = fourier_order_y * 2 + 1
    ky_vector = k0 * (n_I * jnp.sin(theta) * jnp.sin(phi) + fourier_indices_y * (
            wavelength / period[1])).astype(type_complex)

    Kx = jnp.diag(jnp.tile(kx_vector, ff_y).flatten()) / k0
    Ky = jnp.diag(jnp.tile(ky_vector.reshape((-1, 1)), ff_x).flatten()) / k0

    resolution_x, resolution_y, resolution_z = resolution
    field_cell = jnp.zeros((resolution_z * len(layer_info_list), resolution_y, resolution_x, 6), dtype=type_complex)

    T_layer = T1

    big_I = jnp.eye((len(T1))).astype(type_complex)

    # From the first layer
    for idx_layer, (E_conv_i, q, W_11, W_12, W_21, W_22, V_11, V_12, V_21, V_22, big_X, big_A_i, big_B, d)\
            in enumerate(layer_info_list[::-1]):
        c = jnp.block([[big_I], [big_B @ big_A_i @ big_X]]) @ T_layer

        ff = len(c) // 4

        c1_plus = c[0 * ff:1 * ff]
        c2_plus = c[1 * ff:2 * ff]
        c1_minus = c[2 * ff:3 * ff]
        c2_minus = c[3 * ff:4 * ff]

        q1 = q[:len(q) // 2]
        q2 = q[len(q) // 2:]
        big_Q1 = jnp.diag(q1)
        big_Q2 = jnp.diag(q2)

        for k in range(resolution_z):
            z = k / resolution_z * d

            Sx = W_11 @ (diag_exp(-k0 * big_Q1 * z) @ c1_plus + diag_exp(k0 * big_Q1 * (z - d)) @ c1_minus) \
                 + W_12 @ (diag_exp(-k0 * big_Q2 * z) @ c2_plus + diag_exp(k0 * big_Q2 * (z - d)) @ c2_minus)
            Sy = W_21 @ (diag_exp(-k0 * big_Q1 * z) @ c1_plus + diag_exp(k0 * big_Q1 * (z - d)) @ c1_minus) \
                 + W_22 @ (diag_exp(-k0 * big_Q2 * z) @ c2_plus + diag_exp(k0 * big_Q2 * (z - d)) @ c2_minus)
            Ux = V_11 @ (-diag_exp(-k0 * big_Q1 * z) @ c1_plus + diag_exp(k0 * big_Q1 * (z - d)) @ c1_minus) \
                 + V_12 @ (-diag_exp(-k0 * big_Q2 * z) @ c2_plus + diag_exp(k0 * big_Q2 * (z - d)) @ c2_minus)
            Uy = V_21 @ (-diag_exp(-k0 * big_Q1 * z) @ c1_plus + diag_exp(k0 * big_Q1 * (z - d)) @ c1_minus) \
                 + V_22 @ (-diag_exp(-k0 * big_Q2 * z) @ c2_plus + diag_exp(k0 * big_Q2 * (z - d)) @ c2_minus)
            Sz = -1j * E_conv_i @ (Kx @ Uy - Ky @ Ux)
            Uz = -1j * (Kx @ Sy - Ky @ Sx)

            for j in range(resolution_y):
                y = j * period[1] / resolution_y
                for i in range(resolution_x):
                    x = i * period[0] / resolution_x

                    exp_K = jnp.exp(-1j * kx_vector.reshape((1, -1)) * x) * jnp.exp(
                        -1j * ky_vector.reshape((-1, 1)) * y)
                    exp_K = exp_K.flatten()

                    Ex = Sx.T @ exp_K
                    Ey = Sy.T @ exp_K
                    Ez = Sz.T @ exp_K
                    Hx = -1j * Ux.T @ exp_K
                    Hy = -1j * Uy.T @ exp_K
                    Hz = -1j * Uz.T @ exp_K

                    val = [Ex[0], Ey[0], Ez[0], Hx[0], Hy[0], Hz[0]]

                    field_cell = field_cell.at[resolution_z * idx_layer + k, j, i].set(val)
        T_layer = big_A_i @ big_X @ T_layer

    return field_cell


# def field_dist_2d_lax(wavelength, kx_vector, n_I, theta, phi, fourier_order_x, fourier_order_y, T1, layer_info_list, period,
#                   resolution=(10, 10, 10),
#                   type_complex=jnp.complex128):
#
#     k0 = 2 * jnp.pi / wavelength
#     fourier_indices_y = jnp.arange(-fourier_order_y, fourier_order_y + 1)
#     ff_x = fourier_order_x * 2 + 1
#     ff_y = fourier_order_y * 2 + 1
#     ff_xy = ff_x * ff_y
#     ky_vector = k0 * (n_I * jnp.sin(theta) * jnp.sin(phi) + fourier_indices_y * (
#             wavelength / period[1])).astype(type_complex)
#
#     Kx = jnp.diag(jnp.tile(kx_vector, ff_y).flatten()) / k0
#     Ky = jnp.diag(jnp.tile(ky_vector.reshape((-1, 1)), ff_x).flatten()) / k0
#
#     resolution_x, resolution_y, resolution_z = resolution
#     field_cell = jnp.zeros((resolution_z * len(layer_info_list), resolution_y, resolution_x, 6), dtype=type_complex)
#
#     T_layer = T1
#
#     big_I = jnp.eye((len(T1))).astype(type_complex)
#
#     # From the first layer
#     for idx_layer, (E_conv_i, q, W_11, W_12, W_21, W_22, V_11, V_12, V_21, V_22, big_X, big_A_i, big_B, d)\
#             in enumerate(layer_info_list[::-1]):
#
#         c = jnp.block([[big_I], [big_B @ big_A_i @ big_X]]) @ T_layer
#         i=j=k=0  # delete
#
#         args = [k, j, i, k0, resolution_z, resolution_y, resolution_x, idx_layer, d, kx_vector, ky_vector, q, period, c, Kx, Ky, E_conv_i, W_11, W_12, W_21, W_22, V_11, V_12, V_21]
#
#         res_size = 1 * 9 + ff_x + ff_y + 2*ff_xy + 2 + 2*2*ff_xy + 11 * ff_xy**2
#         res = jnp.zeros(res_size, dtype=type_complex)
#
#         b = 0
#         for ix, item in enumerate(args):
#             if type(item) in (float, int):
#                 length = 1
#                 val = item
#             elif isinstance(item, jax.numpy.ndarray):
#                 length = item.size
#                 val = item.flatten()
#             elif type(item) is list:
#                 length = len(item)
#                 val = jnp.array(item)
#             else:
#                 raise
#
#             res[b:b+length] = val
#             b += length
#
#         ress = jnp.tile(res, (resolution_z * resolution_y * resolution_x, 1))
#
#         base = jnp.arange(resolution_x)
#         i_list = jnp.tile(base, resolution_z * resolution_y)[:,None]
#         j_list = jnp.tile(jnp.repeat(base, resolution_x), resolution_z)[:,None]
#         k_list = jnp.repeat(base, resolution_x * resolution_y)[:,None]
#         kji = jnp.hstack((k_list, j_list, i_list))
#
#         resss = ress.at[:, :3].set(kji)
#
#         def calc(field_cell, args):
#             k = args[0]
#             j = args[1]
#             i = args[2]
#             k0 = args[3]
#             resolution_z = args[4]
#             resolution_y = args[5]
#             resolution_x = args[6]
#             idx_layer = args[7]
#             d = args[8]
#
#             b, e = 9, 9 + ff_x
#             kx_vector = args[b:e]
#             b, e = e, e + ff_y
#             ky_vector = args[b:e]
#             b, e = e, e + 2 * ff_xy
#             q = args[b:e]
#             b, e = e, e + 2
#             period = args[b:e]
#
#             b, e = e, e + 2 * 2 * ff_xy
#             c = args[b:e].reshape((-1, 1))
#
#             b, e = e, e + ff_xy * ff_xy
#             Kx = args[b:e].reshape((ff_xy, ff_xy))
#             b, e = e, e + ff_xy * ff_xy
#             Ky = args[b:e].reshape((ff_xy, ff_xy))
#             b, e = e, e + ff_xy * ff_xy
#             E_conv_i = args[b:e].reshape((ff_xy, ff_xy))
#             b, e = e, e + ff_xy * ff_xy
#             W_11 = args[b:e].reshape((ff_xy, ff_xy))
#             b, e = e, e + ff_xy * ff_xy
#             W_12 = args[b:e].reshape((ff_xy, ff_xy))
#             b, e = e, e + ff_xy * ff_xy
#             W_21 = args[b:e].reshape((ff_xy, ff_xy))
#             b, e = e, e + ff_xy * ff_xy
#             W_22 = args[b:e].reshape((ff_xy, ff_xy))
#             b, e = e, e + ff_xy * ff_xy
#             V_11 = args[b:e].reshape((ff_xy, ff_xy))
#             b, e = e, e + ff_xy * ff_xy
#             V_12 = args[b:e].reshape((ff_xy, ff_xy))
#             b, e = e, e + ff_xy * ff_xy
#             V_21 = args[b:e].reshape((ff_xy, ff_xy))
#             b, e = e, e + ff_xy * ff_xy
#             V_22 = args[b:e].reshape((ff_xy, ff_xy))
#
#             y = j * period[1] / resolution_y
#             Sx, Sy, Ux, Uy, Sz, Uz = z_loop_2d(k, c, k0, Kx, Ky, resolution_z, E_conv_i, q, W_11, W_12, W_21, W_22,
#                                                V_11, V_12, V_21, V_22, d)
#             val = x_loop_2d(period, resolution_x, kx_vector, ky_vector, Sx, Sy, Sz, Ux, Uy, Uz, y, i)
#             field_cell = field_cell.at[(resolution_z * idx_layer + k).real.astype(int), j.real.astype(int), i.real.astype(int)].set(val)
#
#             return field_cell, val
#
#         field_cell, _ = jax.lax.scan(calc, field_cell, resss)
#     return field_cell
#
#
#
# def field_dist_2d_lax_heavy(wavelength, kx_vector, n_I, theta, phi, fourier_order_x, fourier_order_y, T1, layer_info_list, period,
#                   resolution=(10, 10, 10),
#                   type_complex=jnp.complex128):
#
#     k0 = 2 * jnp.pi / wavelength
#     fourier_indices_y = jnp.arange(-fourier_order_y, fourier_order_y + 1)
#     ff_x = fourier_order_x * 2 + 1
#     ff_y = fourier_order_y * 2 + 1
#     ff_xy = ff_x * ff_y
#     ky_vector = k0 * (n_I * jnp.sin(theta) * jnp.sin(phi) + fourier_indices_y * (
#             wavelength / period[1])).astype(type_complex)
#
#     Kx = jnp.diag(jnp.tile(kx_vector, ff_y).flatten()) / k0
#     Ky = jnp.diag(jnp.tile(ky_vector.reshape((-1, 1)), ff_x).flatten()) / k0
#
#     resolution_x, resolution_y, resolution_z = resolution
#     field_cell = jnp.zeros((resolution_z * len(layer_info_list), resolution_y, resolution_x, 6), dtype=type_complex)
#
#     T_layer = T1
#
#     big_I = jnp.eye((len(T1))).astype(type_complex)
#
#     # From the first layer
#     for idx_layer, (E_conv_i, q, W_11, W_12, W_21, W_22, V_11, V_12, V_21, V_22, big_X, big_A_i, big_B, d)\
#             in enumerate(layer_info_list[::-1]):
#
#         c = jnp.block([[big_I], [big_B @ big_A_i @ big_X]]) @ T_layer
#         i=j=k=0  # delete
#
#         args = [k, j, i, k0, resolution_z, resolution_y, resolution_x, idx_layer, d, kx_vector, ky_vector, q, period, c, Kx, Ky, E_conv_i, W_11, W_12, W_21, W_22, V_11, V_12, V_21]
#
#         res_size = 1 * 9 + ff_x + ff_y + 2*ff_xy + 2 + 2*2*ff_xy + 11 * ff_xy**2
#         res = jnp.zeros(res_size, dtype=type_complex)
#
#         b = 0
#         for ix, item in enumerate(args):
#             if type(item) in (float, int):
#                 length = 1
#                 val = item
#             elif isinstance(item, jax.numpy.ndarray):
#                 length = item.size
#                 val = item.flatten()
#             elif type(item) is list:
#                 length = len(item)
#                 val = jnp.array(item)
#             else:
#                 raise
#
#             res[b:b+length] = val
#             b += length
#
#         ress = jnp.tile(res, (resolution_z * resolution_y * resolution_x, 1))
#
#         base = jnp.arange(resolution_x)
#         i_list = jnp.tile(base, resolution_z * resolution_y)[:,None]
#         j_list = jnp.tile(jnp.repeat(base, resolution_x), resolution_z)[:,None]
#         k_list = jnp.repeat(base, resolution_x * resolution_y)[:,None]
#         kji = jnp.hstack((k_list, j_list, i_list))
#
#         resss = ress.at[:, :3].set(kji)
#
#         def calc(field_cell, args):
#             k = args[0]
#             j = args[1]
#             i = args[2]
#             k0 = args[3]
#             resolution_z = args[4]
#             resolution_y = args[5]
#             resolution_x = args[6]
#             idx_layer = args[7]
#             d = args[8]
#
#             b, e = 9, 9 + ff_x
#             kx_vector = args[b:e]
#             b, e = e, e + ff_y
#             ky_vector = args[b:e]
#             b, e = e, e + 2 * ff_xy
#             q = args[b:e]
#             b, e = e, e + 2
#             period = args[b:e]
#
#             b, e = e, e + 2 * 2 * ff_xy
#             c = args[b:e].reshape((-1, 1))
#
#             b, e = e, e + ff_xy * ff_xy
#             Kx = args[b:e].reshape((ff_xy, ff_xy))
#             b, e = e, e + ff_xy * ff_xy
#             Ky = args[b:e].reshape((ff_xy, ff_xy))
#             b, e = e, e + ff_xy * ff_xy
#             E_conv_i = args[b:e].reshape((ff_xy, ff_xy))
#             b, e = e, e + ff_xy * ff_xy
#             W_11 = args[b:e].reshape((ff_xy, ff_xy))
#             b, e = e, e + ff_xy * ff_xy
#             W_12 = args[b:e].reshape((ff_xy, ff_xy))
#             b, e = e, e + ff_xy * ff_xy
#             W_21 = args[b:e].reshape((ff_xy, ff_xy))
#             b, e = e, e + ff_xy * ff_xy
#             W_22 = args[b:e].reshape((ff_xy, ff_xy))
#             b, e = e, e + ff_xy * ff_xy
#             V_11 = args[b:e].reshape((ff_xy, ff_xy))
#             b, e = e, e + ff_xy * ff_xy
#             V_12 = args[b:e].reshape((ff_xy, ff_xy))
#             b, e = e, e + ff_xy * ff_xy
#             V_21 = args[b:e].reshape((ff_xy, ff_xy))
#             b, e = e, e + ff_xy * ff_xy
#             V_22 = args[b:e].reshape((ff_xy, ff_xy))
#
#             y = j * period[1] / resolution_y
#             Sx, Sy, Ux, Uy, Sz, Uz = z_loop_2d(k, c, k0, Kx, Ky, resolution_z, E_conv_i, q, W_11, W_12, W_21, W_22,
#                                                V_11, V_12, V_21, V_22, d)
#             val = x_loop_2d(period, resolution_x, kx_vector, ky_vector, Sx, Sy, Sz, Ux, Uy, Uz, y, i)
#             field_cell = field_cell.at[(resolution_z * idx_layer + k).real.astype(int), j.real.astype(int), i.real.astype(int)].set(val)
#
#             return field_cell, val
#
#         field_cell, _ = jax.lax.scan(calc, field_cell, resss)
#     return field_cell


def field_plot(field_cell, pol=0, plot_indices=(1, 1, 1, 1, 1, 1), y_slice=0, z_slice=-1, zx=True, yx=True):
    try:
        import matplotlib.pyplot as plt
    except (ImportError, ModuleNotFoundError) as e:
        print(e)
        print('To use field_plot(), please install matplotlib')
        raise e

    if field_cell.shape[-1] == 6:  # 2D grating
        title = ['2D Ex', '2D Ey', '2D Ez', '2D Hx', '2D Hy', '2D Hz', ]
    else:  # 1D grating
        if pol == 0:  # TE
            title = ['1D Ey', '1D Hx', '1D Hz', ]
        else:  # TM
            title = ['1D Hy', '1D Ex', '1D Ez', ]

    if zx:
        for idx in range(len(title)):
            if plot_indices[idx]:
                plt.imshow((abs(field_cell[:, y_slice, :, idx]) ** 2), cmap='jet', aspect='auto')
                # plt.clim(0, 2)  # identical to caxis([-4,4]) in MATLAB
                plt.colorbar()
                plt.title(title[idx])
                plt.show()
    if yx:
        for idx in range(len(title)):
            if plot_indices[idx]:
                plt.imshow((abs(field_cell[z_slice, :, :, idx]) ** 2), cmap='jet', aspect='auto')
                # plt.clim(0, 3.5)  # identical to caxis([-4,4]) in MATLAB
                plt.colorbar()
                plt.title(title[idx])
                plt.show()


def diag_exp(x):
    return jnp.diag(jnp.exp(jnp.diag(x)))


def diag_exp_batch(x):
    res = jnp.zeros(x.shape, dtype=x.dtype)
    ix = jnp.diag_indices_from(x[0])
    res = res.at[:, ix[0], ix[1]].set(jnp.exp(x[:, ix[0], ix[1]]))
    return res
