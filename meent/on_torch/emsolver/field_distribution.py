import torch


def field_dist_1d_vectorized_ji(wavelength, kx_vector, T1, layer_info_list, period,
                                pol, res_x=20, res_y=20, res_z=20,  device='cpu',
                                type_complex=torch.complex128, type_float=torch.float64):

    k0 = 2 * torch.pi / wavelength
    Kx = torch.diag(kx_vector / k0)

    field_cell = torch.zeros((res_z * len(layer_info_list), res_y, res_x, 3), dtype=type_complex)

    T_layer = T1

    # From the first layer
    for idx_layer, (E_conv_i, q, W, X, a_i, b, d) in enumerate(layer_info_list[::-1]):

        c1 = T_layer[:, None]
        c2 = b @ a_i @ X @ T_layer[:, None]

        Q = torch.diag(q)

        if pol == 0:
            V = W @ Q
            EKx = None

        else:
            V = E_conv_i @ W @ Q
            EKx = E_conv_i @ Kx

        for k in range(res_z):
            z = k / res_z * d

            if pol == 0:
                Sy = W @ (diag_exp(-k0 * Q * z) @ c1 + diag_exp(k0 * Q * (z - d)) @ c2)
                Ux = V @ (-diag_exp(-k0 * Q * z) @ c1 + diag_exp(k0 * Q * (z - d)) @ c2)
                C = Kx @ Sy

                x_1d = torch.arange(res_x, dtype=type_float, device=device).reshape((1, -1, 1))
                x_1d = -1j * x_1d * period[0] / res_x
                x_2d = torch.tile(x_1d, (res_y, 1, 1))
                x_2d = x_2d * kx_vector
                x_2d = x_2d.reshape((res_y, res_x, 1, len(kx_vector)))

                exp_K = torch.exp(x_2d)
                exp_K = exp_K.reshape((res_y, res_x, -1))

                Ey = exp_K @ Sy
                Hx = -1j * exp_K @ Ux
                Hz = -1j * exp_K @ C

                val = torch.cat((Ey, Hx, Hz), -1)

            else:
                Uy = W @ (diag_exp(-k0 * Q * z) @ c1 + diag_exp(k0 * Q * (z - d)) @ c2)
                Sx = V @ (-diag_exp(-k0 * Q * z) @ c1 + diag_exp(k0 * Q * (z - d)) @ c2)

                C = EKx @ Uy  # there is a better option for convergence

                x_1d = torch.arange(res_x, dtype=type_float, device=device).reshape((1, -1, 1))
                x_1d = -1j * x_1d * period[0] / res_x
                x_2d = torch.tile(x_1d, (res_y, 1, 1))
                x_2d = x_2d * kx_vector
                x_2d = x_2d.reshape((res_y, res_x, 1, len(kx_vector)))

                exp_K = torch.exp(x_2d)
                exp_K = exp_K.reshape((res_y, res_x, -1))

                Hy = exp_K @ Uy
                Ex = 1j * exp_K @ Sx
                Ez = -1j * exp_K @ C

                val = torch.cat((Hy, Ex, Ez), -1)

            field_cell[res_z * idx_layer + k] = val

        T_layer = a_i @ X @ T_layer

    return field_cell


def field_dist_1d_conical_vectorized_ji(wavelength, kx_vector, n_I, theta, phi, T1, layer_info_list, period,
                                        res_x=20, res_y=20, res_z=20, device='cpu',
                                        type_complex=torch.complex128, type_float=torch.float64):

    k0 = 2 * torch.pi / wavelength
    ky = k0 * n_I * torch.sin(theta) * torch.sin(phi)
    Kx = torch.diag(kx_vector / k0)

    field_cell = torch.zeros((res_z * len(layer_info_list), res_y, res_x, 6), dtype=type_complex)

    T_layer = T1

    big_I = torch.eye((len(T1)), device=device).type(type_complex)

    # From the first layer
    for idx_layer, [E_conv_i, q_1, q_2, W_1, W_2, V_11, V_12, V_21, V_22, big_X, big_A_i, big_B, d] \
            in enumerate(layer_info_list[::-1]):

        c = torch.cat([big_I, big_B @ big_A_i @ big_X])  @ T_layer

        for k in range(res_z):
            Sx, Sy, Ux, Uy, Sz, Uz = z_loop_1d_conical(k, c, k0, Kx, ky, res_z, E_conv_i, q_1, q_2, W_1, W_2, V_11, V_12, V_21, V_22, d)

            x_1d = torch.arange(res_x, dtype=type_float, device=device).reshape((1, -1, 1))
            x_1d = -1j * x_1d * period[0] / res_x
            x_2d = torch.tile(x_1d, (res_y, 1, 1))
            x_2d = x_2d * kx_vector
            x_2d = x_2d.reshape((res_y, res_x, 1, len(kx_vector)))

            exp_K = torch.exp(x_2d)
            exp_K = exp_K.reshape((res_y, res_x, -1))

            Ex = exp_K @ Sx
            Ey = exp_K @ Sy
            Ez = exp_K @ Sz

            Hx = -1j * exp_K @ Ux
            Hy = -1j * exp_K @ Uy
            Hz = -1j * exp_K @ Uz

            val = torch.stack((Ex, Ey, Ez, Hx, Hy, Hz), -1)

            field_cell[res_z * idx_layer + k] = val

        T_layer = big_A_i @ big_X @ T_layer

    return field_cell


def field_dist_2d_vectorized_ji(wavelength, kx_vector, n_I, theta, phi, fourier_order_x, fourier_order_y, T1, layer_info_list, period,
                                res_x=20, res_y=20, res_z=20, device='cpu', type_complex=torch.complex128, type_float=torch.float64):

    k0 = 2 * torch.pi / wavelength

    fourier_indices_y = torch.arange(-fourier_order_y, fourier_order_y + 1, dtype=type_float, device=device)
    ff_x = fourier_order_x * 2 + 1
    ff_y = fourier_order_y * 2 + 1
    ky_vector = k0 * (n_I * torch.sin(theta) * torch.sin(phi) + fourier_indices_y * (
            wavelength / period[1])).type(type_complex)

    Kx = torch.diag(torch.tile(kx_vector, (ff_y, )).flatten()) / k0
    Ky = torch.diag(torch.tile(ky_vector.reshape((-1, 1)), (ff_x, )).flatten()) / k0

    field_cell = torch.zeros((res_z * len(layer_info_list), res_y, res_x, 6), dtype=type_complex)

    T_layer = T1

    big_I = torch.eye((len(T1)), device=device).type(type_complex)

    # From the first layer
    for idx_layer, (E_conv_i, q, W_11, W_12, W_21, W_22, V_11, V_12, V_21, V_22, big_X, big_A_i, big_B, d)\
            in enumerate(layer_info_list[::-1]):

        c = torch.cat([big_I, big_B @ big_A_i @ big_X])  @ T_layer

        for k in range(res_z):
            Sx, Sy, Ux, Uy, Sz, Uz = z_loop_2d(k, c, k0, Kx, Ky, res_z, E_conv_i, q, W_11, W_12, W_21, W_22, V_11, V_12, V_21, V_22, d)

            x_1d = torch.arange(res_x, dtype=type_float, device=device).reshape((1, -1, 1))
            y_1d = torch.arange(res_y, dtype=type_float, device=device).reshape((-1, 1, 1))

            x_1d = -1j * x_1d * period[0] / res_x
            y_1d = -1j * y_1d * period[1] / res_y

            x_2d = torch.tile(x_1d, (res_y, 1, 1))
            y_2d = torch.tile(y_1d, (1, res_x, 1))

            x_2d = x_2d * kx_vector
            y_2d = y_2d * ky_vector

            x_2d = x_2d.reshape((res_y, res_x, 1, len(kx_vector)))
            y_2d = y_2d.reshape((res_y, res_x, len(ky_vector), 1))

            exp_K = torch.exp(x_2d) * torch.exp(y_2d)
            exp_K = exp_K.reshape((res_y, res_x, -1))

            Ex = exp_K @ Sx
            Ey = exp_K @ Sy
            Ez = exp_K @ Sz

            Hx = -1j * exp_K @ Ux
            Hy = -1j * exp_K @ Uy
            Hz = -1j * exp_K @ Uz

            val = torch.stack((Ex.squeeze(), Ey.squeeze(), Ez.squeeze(), Hx.squeeze(), Hy.squeeze(), Hz.squeeze()), -1)

            field_cell[res_z * idx_layer + k] = val

        T_layer = big_A_i @ big_X @ T_layer

    return field_cell


def field_dist_1d_vectorized_kji(wavelength, kx_vector, T1, layer_info_list, period,
                                 pol, res_x=20, res_y=20, res_z=20,  device='cpu',
                                 type_complex=torch.complex128, type_float=torch.float64):

    k0 = 2 * torch.pi / wavelength
    Kx = torch.diag(kx_vector / k0)

    field_cell = torch.zeros((res_z * len(layer_info_list), res_y, res_x, 3), dtype=type_complex)

    T_layer = T1

    # From the first layer
    for idx_layer, (E_conv_i, q, W, X, a_i, b, d) in enumerate(layer_info_list[::-1]):

        c1 = T_layer[:, None]
        c2 = b @ a_i @ X @ T_layer[:, None]

        Q = torch.diag(q)

        if pol == 0:
            V = W @ Q
            EKx = None

        else:
            V = E_conv_i @ W @ Q
            EKx = E_conv_i @ Kx

        z_1d = torch.arange(res_z, dtype=type_float, device=device).reshape((-1, 1, 1)) / res_z * d

        if pol == 0:
            Sy = W @ (diag_exp_batch(-k0 * Q * z_1d) @ c1 + diag_exp_batch(k0 * Q * (z_1d - d)) @ c2)
            Ux = V @ (-diag_exp_batch(-k0 * Q * z_1d) @ c1 + diag_exp_batch(k0 * Q * (z_1d - d)) @ c2)
            C = Kx @ Sy

            x_1d = torch.arange(res_x, dtype=type_float, device=device).reshape((1, -1, 1))
            x_1d = -1j * x_1d * period[0] / res_x
            x_2d = torch.tile(x_1d, (res_y, 1, 1))
            x_2d = x_2d * kx_vector
            x_2d = x_2d.reshape((res_y, res_x, 1, len(kx_vector)))

            exp_K = torch.exp(x_2d)
            exp_K = exp_K.reshape((res_y, res_x, -1))

            Ey = exp_K[:, :, None, :] @ Sy[:, None, None, :, :]
            Hx = -1j * exp_K[:, :, None, :] @ Ux[:, None, None, :, :]
            Hz = -1j * exp_K[:, :, None, :] @ C[:, None, None, :, :]

            val = torch.cat((Ey.squeeze(-1), Hx.squeeze(-1), Hz.squeeze(-1)), -1)

        else:
            Uy = W @ (diag_exp_batch(-k0 * Q * z_1d) @ c1 + diag_exp_batch(k0 * Q * (z_1d - d)) @ c2)
            Sx = V @ (-diag_exp_batch(-k0 * Q * z_1d) @ c1 + diag_exp_batch(k0 * Q * (z_1d - d)) @ c2)

            C = EKx @ Uy  # there is a better option for convergence

            x_1d = torch.arange(res_x, dtype=type_float, device=device).reshape((1, -1, 1))
            x_1d = -1j * x_1d * period[0] / res_x
            x_2d = torch.tile(x_1d, (res_y, 1, 1))
            x_2d = x_2d * kx_vector
            x_2d = x_2d.reshape((res_y, res_x, 1, len(kx_vector)))

            exp_K = torch.exp(x_2d)
            exp_K = exp_K.reshape((res_y, res_x, -1))

            Hy = exp_K[:, :, None, :] @ Uy[:, None, None, :, :]
            Ex = 1j * exp_K[:, :, None, :] @ Sx[:, None, None, :, :]
            Ez = -1j * exp_K[:, :, None, :] @ C[:, None, None, :, :]

            val = torch.cat((Hy.squeeze(-1), Ex.squeeze(-1), Ez.squeeze(-1)), -1)

        field_cell[res_z * idx_layer:res_z * (idx_layer + 1)] = val

        T_layer = a_i @ X @ T_layer

    return field_cell


def field_dist_1d_conical_vectorized_kji(wavelength, kx_vector, n_I, theta, phi, T1, layer_info_list, period,
                                         res_x=20, res_y=20, res_z=20, device='cpu', type_complex=torch.complex128, type_float=torch.float64):

    k0 = 2 * torch.pi / wavelength
    ky = k0 * n_I * torch.sin(theta) * torch.sin(phi)
    Kx = torch.diag(kx_vector / k0)

    field_cell = torch.zeros((res_z * len(layer_info_list), res_y, res_x, 6), dtype=type_complex)

    T_layer = T1[:, None]

    big_I = torch.eye((len(T1)), device=device).type(type_complex)

    # From the first layer
    for idx_layer, [E_conv_i, q_1, q_2, W_1, W_2, V_11, V_12, V_21, V_22, big_X, big_A_i, big_B, d] \
            in enumerate(layer_info_list[::-1]):

        c = torch.cat([big_I, big_B @ big_A_i @ big_X])  @ T_layer

        z_1d = torch.arange(res_z, dtype=type_float, device=device).reshape((-1, 1, 1)) / res_z * d

        ff = len(c) // 4

        c1_plus = c[0 * ff:1 * ff]
        c2_plus = c[1 * ff:2 * ff]
        c1_minus = c[2 * ff:3 * ff]
        c2_minus = c[3 * ff:4 * ff]

        big_Q1 = torch.diag(q_1)
        big_Q2 = torch.diag(q_2)

        Sx = W_2 @ (diag_exp_batch(-k0 * big_Q2 * z_1d) @ c2_plus + diag_exp_batch(k0 * big_Q2 * (z_1d - d)) @ c2_minus)

        Sy = V_11 @ (diag_exp_batch(-k0 * big_Q1 * z_1d) @ c1_plus + diag_exp_batch(k0 * big_Q1 * (z_1d - d)) @ c1_minus) \
             + V_12 @ (diag_exp_batch(-k0 * big_Q2 * z_1d) @ c2_plus + diag_exp_batch(k0 * big_Q2 * (z_1d - d)) @ c2_minus)

        Ux = W_1 @ (-diag_exp_batch(-k0 * big_Q1 * z_1d) @ c1_plus + diag_exp_batch(k0 * big_Q1 * (z_1d - d)) @ c1_minus)

        Uy = V_21 @ (-diag_exp_batch(-k0 * big_Q1 * z_1d) @ c1_plus + diag_exp_batch(k0 * big_Q1 * (z_1d - d)) @ c1_minus) \
             + V_22 @ (-diag_exp_batch(-k0 * big_Q2 * z_1d) @ c2_plus + diag_exp_batch(k0 * big_Q2 * (z_1d - d)) @ c2_minus)

        Sz = -1j * E_conv_i @ (Kx @ Uy - ky * Ux)
        Uz = -1j * (Kx @ Sy - ky * Sx)

        x_1d = torch.arange(res_x, dtype=type_float, device=device).reshape((1, -1, 1))
        x_1d = -1j * x_1d * period[0] / res_x
        x_2d = torch.tile(x_1d, (res_y, 1, 1))
        x_2d = x_2d * kx_vector
        x_2d = x_2d.reshape((res_y, res_x, 1, len(kx_vector)))

        exp_K = torch.exp(x_2d)
        exp_K = exp_K.reshape((res_y, res_x, -1))

        Ex = exp_K[:, :, None, :] @ Sx[:, None, None, :, :]
        Ey = exp_K[:, :, None, :] @ Sy[:, None, None, :, :]
        Ez = exp_K[:, :, None, :] @ Sz[:, None, None, :, :]

        Hx = -1j * exp_K[:, :, None, :] @ Ux[:, None, None, :, :]
        Hy = -1j * exp_K[:, :, None, :] @ Uy[:, None, None, :, :]
        Hz = -1j * exp_K[:, :, None, :] @ Uz[:, None, None, :, :]

        val = torch.cat(
            (Ex.squeeze(-1), Ey.squeeze(-1), Ez.squeeze(-1), Hx.squeeze(-1), Hy.squeeze(-1), Hz.squeeze(-1)), -1)
        field_cell[res_z * idx_layer:res_z * (idx_layer + 1)] = val
        T_layer = big_A_i @ big_X @ T_layer

    return field_cell


def field_dist_2d_vectorized_kji(wavelength, kx_vector, n_I, theta, phi, fourier_order_x, fourier_order_y, T1, layer_info_list, period,
                                 res_x=20, res_y=20, res_z=20, device='cpu', type_complex=torch.complex128, type_float=torch.float64):

    k0 = 2 * torch.pi / wavelength

    fourier_indices_y = torch.arange(-fourier_order_y, fourier_order_y + 1, dtype=type_float, device=device)
    ff_x = fourier_order_x * 2 + 1
    ff_y = fourier_order_y * 2 + 1
    ky_vector = k0 * (n_I * torch.sin(theta) * torch.sin(phi) + fourier_indices_y * (
            wavelength / period[1])).type(type_complex)

    Kx = torch.diag(torch.tile(kx_vector, (ff_y, )).flatten()) / k0
    Ky = torch.diag(torch.tile(ky_vector.reshape((-1, 1)), (ff_x, )).flatten()) / k0

    field_cell = torch.zeros((res_z * len(layer_info_list), res_y, res_x, 6), dtype=type_complex)

    T_layer = T1

    big_I = torch.eye((len(T1)), device=device).type(type_complex)

    # From the first layer
    for idx_layer, (E_conv_i, q, W_11, W_12, W_21, W_22, V_11, V_12, V_21, V_22, big_X, big_A_i, big_B, d)\
            in enumerate(layer_info_list[::-1]):

        c = torch.cat([big_I, big_B @ big_A_i @ big_X])  @ T_layer
        z_1d = torch.arange(res_z, dtype=type_float, device=device).reshape((-1, 1, 1)) / res_z * d

        ff = len(c) // 4

        c1_plus = c[0 * ff:1 * ff]
        c2_plus = c[1 * ff:2 * ff]
        c1_minus = c[2 * ff:3 * ff]
        c2_minus = c[3 * ff:4 * ff]

        q1 = q[:len(q) // 2]
        q2 = q[len(q) // 2:]
        big_Q1 = torch.diag(q1)
        big_Q2 = torch.diag(q2)
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

        x_1d = torch.arange(res_x, dtype=type_float, device=device).reshape((1, -1, 1))
        y_1d = torch.arange(res_y, dtype=type_float, device=device).reshape((-1, 1, 1))

        x_1d = -1j * x_1d * period[0] / res_x
        y_1d = -1j * y_1d * period[1] / res_y

        x_2d = torch.tile(x_1d, (res_y, 1, 1))
        y_2d = torch.tile(y_1d, (1, res_x, 1))

        x_2d = x_2d * kx_vector
        y_2d = y_2d * ky_vector

        x_2d = x_2d.reshape((res_y, res_x, 1, len(kx_vector)))
        y_2d = y_2d.reshape((res_y, res_x, len(ky_vector), 1))

        exp_K = torch.exp(x_2d) * torch.exp(y_2d)
        exp_K = exp_K.reshape((res_y, res_x, -1))

        Ex = exp_K[:, :, None, :] @ Sx[:, None, None, :, :]
        Ey = exp_K[:, :, None, :] @ Sy[:, None, None, :, :]
        Ez = exp_K[:, :, None, :] @ Sz[:, None, None, :, :]

        Hx = -1j * exp_K[:, :, None, :] @ Ux[:, None, None, :, :]
        Hy = -1j * exp_K[:, :, None, :] @ Uy[:, None, None, :, :]
        Hz = -1j * exp_K[:, :, None, :] @ Uz[:, None, None, :, :]

        val = torch.cat(
            (Ex.squeeze(-1), Ey.squeeze(-1), Ez.squeeze(-1), Hx.squeeze(-1), Hy.squeeze(-1), Hz.squeeze(-1)), -1)

        field_cell[res_z * idx_layer:res_z * (idx_layer + 1)] = val

        T_layer = big_A_i @ big_X @ T_layer

    return field_cell


def field_dist_1d_vanilla(wavelength, kx_vector, T1, layer_info_list, period, pol, res_x=20, res_y=20, res_z=20,
                          device='cpu', type_complex=torch.complex128, *args, **kwargs):

    k0 = 2 * torch.pi / wavelength
    Kx = torch.diag(kx_vector / k0)

    field_cell = torch.zeros((res_z * len(layer_info_list), res_y, res_x, 3)).type(type_complex)

    T_layer = T1

    # From the first layer
    for idx_layer, (E_conv_i, q, W, X, a_i, b, d) in enumerate(layer_info_list[::-1]):

        c1 = T_layer[:, None]
        c2 = b @ a_i @ X @ T_layer[:, None]

        Q = torch.diag(q)

        if pol == 0:
            V = W @ Q
            EKx = None

        else:
            V = E_conv_i @ W @ Q
            EKx = E_conv_i @ Kx

        for k in range(res_z):
            z = k / res_z * d

            if pol == 0:  # TE
                Sy = W @ (diag_exp(-k0 * Q * z) @ c1 + diag_exp(k0 * Q * (z - d)) @ c2)
                Ux = V @ (-diag_exp(-k0 * Q * z) @ c1 + diag_exp(k0 * Q * (z - d)) @ c2)
                f_here = (-1j) * Kx @ Sy

                for j in range(res_y):
                    for i in range(res_x):
                        x = i * period[0] / res_x

                        Ey = Sy.T @ torch.exp(-1j * kx_vector.reshape((-1, 1)) * x)
                        Hx = -1j * Ux.T @ torch.exp(-1j * kx_vector.reshape((-1, 1)) * x)
                        Hz = f_here.T @ torch.exp(-1j * kx_vector.reshape((-1, 1)) * x)
                        val = torch.tensor([Ey, Hx, Hz])
                        field_cell[res_z * idx_layer + k, j, i] = val
            else:  # TM
                Uy = W @ (diag_exp(-k0 * Q * z) @ c1 + diag_exp(k0 * Q * (z - d)) @ c2)
                Sx = V @ (-diag_exp(-k0 * Q * z) @ c1 + diag_exp(k0 * Q * (z - d)) @ c2)
                f_here = (-1j) * EKx @ Uy  # there is a better option for convergence

                for j in range(res_y):
                    for i in range(res_x):
                        x = i * period[0] / res_x

                        Hy = Uy.T @ torch.exp(-1j * kx_vector.reshape((-1, 1)) * x)
                        Ex = 1j * Sx.T @ torch.exp(-1j * kx_vector.reshape((-1, 1)) * x)
                        Ez = f_here.T @ torch.exp(-1j * kx_vector.reshape((-1, 1)) * x)
                        val = torch.tensor([Hy, Ex, Ez])
                        field_cell[res_z * idx_layer + k, j, i] = val
        T_layer = a_i @ X @ T_layer

    return field_cell


def field_dist_1d_conical_vanilla(wavelength, kx_vector, n_I, theta, phi, T1, layer_info_list, period, res_x=20, res_y=20, res_z=20,
                                  device='cpu', type_complex=torch.complex128, *args, **kwargs):

    k0 = 2 * torch.pi / wavelength
    ky = k0 * n_I * torch.sin(theta) * torch.sin(phi)
    Kx = torch.diag(kx_vector / k0)

    field_cell = torch.zeros((res_z * len(layer_info_list), res_y, res_x, 6)).type(type_complex)

    T_layer = T1

    big_I = torch.eye((len(T1)), device=device, dtype=type_complex)

    # From the first layer
    for idx_layer, [E_conv_i, q_1, q_2, W_1, W_2, V_11, V_12, V_21, V_22, big_X, big_A_i, big_B, d] \
            in enumerate(layer_info_list[::-1]):

        c = torch.cat([big_I, big_B @ big_A_i @ big_X])  @ T_layer

        cut = len(c) // 4

        c1_plus = c[0*cut:1*cut]
        c2_plus = c[1*cut:2*cut]
        c1_minus = c[2*cut:3*cut]
        c2_minus = c[3*cut:4*cut]

        big_Q1 = torch.diag(q_1)
        big_Q2 = torch.diag(q_2)

        for k in range(res_z):
            z = k / res_z * d

            Sx = W_2 @ (diag_exp(-k0 * big_Q2 * z) @ c2_plus + diag_exp(k0 * big_Q2 * (z - d)) @ c2_minus)

            Sy = V_11 @ (diag_exp(-k0 * big_Q1 * z) @ c1_plus + diag_exp(k0 * big_Q1 * (z - d)) @ c1_minus) \
                 + V_12 @ (diag_exp(-k0 * big_Q2 * z) @ c2_plus + diag_exp(k0 * big_Q2 * (z - d)) @ c2_minus)

            Ux = W_1 @ (-diag_exp(-k0 * big_Q1 * z) @ c1_plus + diag_exp(k0 * big_Q1 * (z - d)) @ c1_minus)

            Uy = V_21 @ (-diag_exp(-k0 * big_Q1 * z) @ c1_plus + diag_exp(k0 * big_Q1 * (z - d)) @ c1_minus) \
                 + V_22 @ (-diag_exp(-k0 * big_Q2 * z) @ c2_plus + diag_exp(k0 * big_Q2 * (z - d)) @ c2_minus)

            Sz = -1j * E_conv_i @ (Kx @ Uy - ky * Ux)

            Uz = -1j * (Kx @ Sy - ky * Sx)

            for j in range(res_y):
                for i in range(res_x):
                    x = i * period[0] / res_x

                    exp_K = torch.exp(-1j*kx_vector.reshape((-1, 1)) * x)

                    Ex = Sx @ exp_K
                    Ey = Sy @ exp_K
                    Ez = Sz @ exp_K

                    Hx = -1j * Ux @ exp_K
                    Hy = -1j * Uy @ exp_K
                    Hz = -1j * Uz @ exp_K

                    field_cell[res_z * idx_layer + k, j, i] = torch.tensor([Ex, Ey, Ez, Hx, Hy, Hz])

        T_layer = big_A_i @ big_X @ T_layer

    return field_cell


def field_dist_2d_vanilla(wavelength, kx_vector, n_I, theta, phi, fourier_order_x, fourier_order_y, T1, layer_info_list,
                          period, res_x=20, res_y=20, res_z=20,
                          device='cpu', type_complex=torch.complex128, type_float=torch.float64):

    k0 = 2 * torch.pi / wavelength

    fourier_indices_y = torch.arange(-fourier_order_y, fourier_order_y + 1, dtype=type_float, device=device)
    ff_x = fourier_order_x * 2 + 1
    ff_y = fourier_order_y * 2 + 1
    ky_vector = k0 * (n_I * torch.sin(theta) * torch.sin(phi) + fourier_indices_y * (
            wavelength / period[1])).type(type_complex)

    Kx = torch.diag(kx_vector.tile(ff_y).flatten()) / k0
    Ky = torch.diag(ky_vector.reshape((-1, 1)).tile(ff_x).flatten() / k0)

    field_cell = torch.zeros((res_z * len(layer_info_list), res_y, res_x, 6)).type(type_complex)

    T_layer = T1

    big_I = torch.eye((len(T1)), device=device, dtype=type_complex)

    # From the first layer
    for idx_layer, (E_conv_i, q, W_11, W_12, W_21, W_22, V_11, V_12, V_21, V_22, big_X, big_A_i, big_B, d)\
            in enumerate(layer_info_list[::-1]):

        c = torch.cat([big_I, big_B @ big_A_i @ big_X])  @ T_layer

        cut = len(c) // 4

        c1_plus = c[0*cut:1*cut]
        c2_plus = c[1*cut:2*cut]
        c1_minus = c[2*cut:3*cut]
        c2_minus = c[3*cut:4*cut]

        q_1 = q[:len(q)//2]
        q_2 = q[len(q)//2:]
        big_Q1 = torch.diag(q_1)
        big_Q2 = torch.diag(q_2)

        for k in range(res_z):
            z = k / res_z * d

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

            for j in range(res_y):
                y = j * period[1] / res_y

                for i in range(res_x):

                    x = i * period[0] / res_x

                    exp_K = torch.exp(-1j*kx_vector.reshape((1, -1)) * x) * torch.exp(-1j*ky_vector.reshape((-1, 1)) * y)
                    exp_K = exp_K.flatten()

                    Ex = Sx.T @ exp_K
                    Ey = Sy.T @ exp_K
                    Ez = Sz.T @ exp_K

                    Hx = -1j * Ux.T @ exp_K
                    Hy = -1j * Uy.T @ exp_K
                    Hz = -1j * Uz.T @ exp_K

                    field_cell[res_z * idx_layer + k, j, i] = torch.tensor([Ex, Ey, Ez, Hx, Hy, Hz])
        T_layer = big_A_i @ big_X @ T_layer

    return field_cell


def field_plot(field_cell, pol=0, plot_indices=(1, 1, 1, 1, 1, 1), y_slice=0, z_slice=-1, zx=True, yx=True):
    try:
        import matplotlib.pyplot as plt
    except (ImportError, ModuleNotFoundError) as e:
        print(e)
        print('To use cal_field(), please install matplotlib')
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
                plt.title(f'{title[idx]}, Side View')
                plt.xlabel('X')
                plt.ylabel('Z')
                plt.show()
    if yx:
        for idx in range(len(title)):
            if plot_indices[idx]:
                plt.imshow((abs(field_cell[z_slice, :, :, idx]) ** 2), cmap='jet', aspect='auto')
                # plt.clim(0, 3.5)  # identical to caxis([-4,4]) in MATLAB
                plt.colorbar()
                plt.title(f'{title[idx]}, Top View')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.show()


def diag_exp(x):
    return torch.diag(torch.exp(torch.diag(x)))


def diag_exp_batch(x):
    res = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
    ix = torch.arange(x.shape[-1], device=x.device)
    res[:, ix, ix] = torch.exp(x[:, ix, ix])
    return res


def z_loop_1d_conical(k, c, k0, Kx, ky, res_z, E_conv_i, q_1, q_2, W_1, W_2, V_11, V_12, V_21, V_22, d):

    z = k / res_z * d

    ff = len(c) // 4

    c1_plus = c[0 * ff:1 * ff]
    c2_plus = c[1 * ff:2 * ff]
    c1_minus = c[2 * ff:3 * ff]
    c2_minus = c[3 * ff:4 * ff]

    big_Q1 = torch.diag(q_1)
    big_Q2 = torch.diag(q_2)

    Sx = W_2 @ (diag_exp(-k0 * big_Q2 * z) @ c2_plus + diag_exp(k0 * big_Q2 * (z - d)) @ c2_minus)

    Sy = V_11 @ (diag_exp(-k0 * big_Q1 * z) @ c1_plus + diag_exp(k0 * big_Q1 * (z - d)) @ c1_minus) \
         + V_12 @ (diag_exp(-k0 * big_Q2 * z) @ c2_plus + diag_exp(k0 * big_Q2 * (z - d)) @ c2_minus)

    Ux = W_1 @ (-diag_exp(-k0 * big_Q1 * z) @ c1_plus + diag_exp(k0 * big_Q1 * (z - d)) @ c1_minus)

    Uy = V_21 @ (-diag_exp(-k0 * big_Q1 * z) @ c1_plus + diag_exp(k0 * big_Q1 * (z - d)) @ c1_minus) \
         + V_22 @ (-diag_exp(-k0 * big_Q2 * z) @ c2_plus + diag_exp(k0 * big_Q2 * (z - d)) @ c2_minus)

    Sz = -1j * E_conv_i @ (Kx @ Uy - ky * Ux)

    Uz = -1j * (Kx @ Sy - ky * Sx)

    return Sx, Sy, Ux, Uy, Sz, Uz


def z_loop_2d(k, c, k0, Kx, Ky, res_z, E_conv_i, q, W_11, W_12, W_21, W_22, V_11, V_12, V_21, V_22, d):

    z = k / res_z * d

    ff = len(c) // 4

    c1_plus = c[0 * ff:1 * ff]
    c2_plus = c[1 * ff:2 * ff]
    c1_minus = c[2 * ff:3 * ff]
    c2_minus = c[3 * ff:4 * ff]

    q1 = q[:len(q) // 2]
    q2 = q[len(q) // 2:]
    big_Q1 = torch.diag(q1)
    big_Q2 = torch.diag(q2)

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

    return Sx, Sy, Ux, Uy, Sz, Uz

