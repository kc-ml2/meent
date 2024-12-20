import torch


def field_dist_1d(wavelength, kx, T1, layer_info_list, period,
                  pol, res_x=20, res_y=20, res_z=20, device='cpu',
                  type_complex=torch.complex128, type_float=torch.float64):
    k0 = 2 * torch.pi / wavelength
    Kx = torch.diag(kx)

    field_cell = torch.zeros((res_z * len(layer_info_list), res_y, res_x, 3), device=device, dtype=type_complex)

    T_layer = T1

    # From the first layer
    for idx_layer, (epz_conv_i, W, V, q, d, A_i, B) in enumerate(layer_info_list[::-1]):

        X = torch.diag(torch.exp(-k0 * q * d))
        c1 = T_layer[:, None]
        c2 = B @ A_i @ X @ T_layer[:, None]
        Q = torch.diag(q)

        z_1d = torch.linspace(0, res_z, res_z, device=device, dtype=type_complex).reshape((-1, 1, 1)) / res_z * d

        My = W @ (d_exp(-k0 * Q * z_1d) @ c1 + d_exp(k0 * Q * (z_1d - d)) @ c2)
        Mx = V @ (-d_exp(-k0 * Q * z_1d) @ c1 + d_exp(k0 * Q * (z_1d - d)) @ c2)
        if pol == 0:
            Mz = -1j * Kx @ My
        else:
            Mz = -1j * epz_conv_i @ Kx @ My if pol else -1j * Kx @ My

        # x_1d = np.arange(1, res_x+1).reshape((1, -1, 1))
        # x_1d = x_1d * period[0] / res_x

        x_1d = torch.linspace(0, period[0], res_x, device=device, dtype=type_complex).reshape((1, -1, 1))

        x_2d = torch.tile(x_1d, (res_y, 1, 1))
        x_2d = x_2d * kx * k0
        x_2d = x_2d.reshape((res_y, res_x, 1, len(kx)))

        inv_fourier = torch.exp(-1j * x_2d)
        inv_fourier = inv_fourier.reshape((res_y, res_x, -1))

        if pol == 0:
            Fy = inv_fourier[:, :, None, :] @ My[:, None, None, :, :]
            Fx = 1j * inv_fourier[:, :, None, :] @ Mx[:, None, None, :, :]
            Fz = 1j * inv_fourier[:, :, None, :] @ Mz[:, None, None, :, :]

        else:
            Fy = inv_fourier[:, :, None, :] @ My[:, None, None, :, :]
            Fx = -1j * inv_fourier[:, :, None, :] @ Mx[:, None, None, :, :]
            Fz = -1j * inv_fourier[:, :, None, :] @ Mz[:, None, None, :, :]

        # val = torch.concatenate((Fy.squeeze(-1), Fx.squeeze(-1), Fz.squeeze(-1)), axis=-1)
        val = torch.cat((Fy.squeeze(-1), Fx.squeeze(-1), Fz.squeeze(-1)), -1)

        field_cell[res_z * idx_layer:res_z * (idx_layer + 1)] = val

        T_layer = A_i @ X @ T_layer

    return field_cell


def field_dist_1d_conical(wavelength, kx, ky, T1, layer_info_list, period,
                          res_x=20, res_y=20, res_z=20, set_field_input=(True, False, False),
                          device='cpu', type_complex=torch.complex128, type_float=torch.float64):
    k0 = 2 * torch.pi / wavelength

    ff_x = len(kx)
    ff_y = len(ky)
    ff_xy = ff_x * ff_y

    Kx = torch.diag(torch.tile(kx, (ff_y,)).flatten())
    Ky = torch.diag(torch.tile(ky.reshape((-1, 1)), (ff_x,)).flatten())

    # field_cell = torch.zeros((res_z * len(layer_info_list), res_y, res_x, 6), device=device, dtype=type_complex)
    field_cell = torch.zeros((sum(set_field_input), res_z * len(layer_info_list), res_y, res_x, 6), device=device,
                             dtype=type_complex)

    # T_layer = T1
    T_layer = T1[list(set_field_input)]

    big_I = torch.eye((len(T1[0])), device=device, dtype=type_complex)
    O = torch.zeros((ff_xy, ff_xy), device=device, dtype=type_complex)

    # From the first layer
    for idx_layer, (epz_conv_i, W, V, q, d, big_A_i, big_B) in enumerate(layer_info_list[::-1]):
        W_1 = W[:, :ff_xy]
        W_2 = W[:, ff_xy:]

        V_11 = V[:ff_xy, :ff_xy]
        V_12 = V[:ff_xy, ff_xy:]
        V_21 = V[ff_xy:, :ff_xy]
        V_22 = V[ff_xy:, ff_xy:]

        q_1 = q[:ff_xy]
        q_2 = q[ff_xy:]

        X_1 = torch.diag(torch.exp(-k0 * q_1 * d))
        X_2 = torch.diag(torch.exp(-k0 * q_2 * d))

        big_X = torch.cat([
            torch.cat([X_1, O], dim=1),
            torch.cat([O, X_2], dim=1)])

        c = torch.cat([big_I, big_B @ big_A_i @ big_X]) @ T_layer

        # z_1d = np.arange(0, res_z, res_z).reshape((-1, 1, 1)) / res_z * d
        z_1d = torch.linspace(0, res_z, res_z, device=device, dtype=type_complex).reshape((-1, 1, 1)) / res_z * d

        # c1_plus = c[0 * ff_xy:1 * ff_xy]
        # c2_plus = c[1 * ff_xy:2 * ff_xy]
        # c1_minus = c[2 * ff_xy:3 * ff_xy]
        # c2_minus = c[3 * ff_xy:4 * ff_xy]
        c1_plus = c[:, 0 * ff_xy:1 * ff_xy]
        c2_plus = c[:, 1 * ff_xy:2 * ff_xy]
        c1_minus = c[:, 2 * ff_xy:3 * ff_xy]
        c2_minus = c[:, 3 * ff_xy:4 * ff_xy]

        big_Q1 = torch.diag(q_1)
        big_Q2 = torch.diag(q_2)

        Sx = W_2 @ (d_exp(-k0 * big_Q2 * z_1d)[:, None, :, :] @ c2_plus
                    + d_exp(k0 * big_Q2 * (z_1d - d))[:, None, :, :] @ c2_minus)
        Sy = V_11 @ (d_exp(-k0 * big_Q1 * z_1d)[:, None, :, :] @ c1_plus
                     + d_exp(k0 * big_Q1 * (z_1d - d))[:, None, :, :] @ c1_minus) \
             + V_12 @ (d_exp(-k0 * big_Q2 * z_1d)[:, None, :, :] @ c2_plus
                       + d_exp(k0 * big_Q2 * (z_1d - d))[:, None, :, :] @ c2_minus)
        Ux = W_1 @ (-d_exp(-k0 * big_Q1 * z_1d)[:, None, :, :] @ c1_plus
                    + d_exp(k0 * big_Q1 * (z_1d - d))[:, None, :, :] @ c1_minus)
        Uy = V_21 @ (-d_exp(-k0 * big_Q1 * z_1d)[:, None, :, :] @ c1_plus
                     + d_exp(k0 * big_Q1 * (z_1d - d))[:, None, :, :] @ c1_minus) \
             + V_22 @ (-d_exp(-k0 * big_Q2 * z_1d)[:, None, :, :] @ c2_plus
                       + d_exp(k0 * big_Q2 * (z_1d - d))[:, None, :, :] @ c2_minus)
        Sz = -1j * epz_conv_i @ (Kx @ Uy - Ky @ Ux)
        Uz = -1j * (Kx @ Sy - Ky @ Sx)

        # x_1d = np.arange(res_x).reshape((1, -1, 1)) * period[0] / res_x
        x_1d = torch.linspace(0, period[0], res_x, device=device, dtype=type_complex).reshape((1, -1, 1))
        x_2d = torch.tile(x_1d, (res_y, 1, 1))
        x_2d = x_2d * kx * k0
        x_2d = x_2d.reshape((res_y, res_x, 1, len(kx)))

        # y_1d = np.arange(res_y-1, -1, -1).reshape((-1, 1, 1)) * period[1] / res_y
        # y_1d = torch.linspace(0, period[1], res_y, device=device, dtype=type_complex)[::-1].reshape((-1, 1, 1))
        y_1d = torch.flip(torch.linspace(0, period[1], res_y, device=device, dtype=type_complex), dims=(0,)).reshape(
            (-1, 1, 1))
        y_2d = torch.tile(y_1d, (1, res_x, 1))
        y_2d = y_2d * ky * k0
        y_2d = y_2d.reshape((res_y, res_x, len(ky), 1))

        inv_fourier = torch.exp(-1j * x_2d) * torch.exp(-1j * y_2d)
        inv_fourier = inv_fourier.reshape((res_y, res_x, -1))

        Ex = inv_fourier[:, :, None, :] @ Sx[:, :, None, None, :, :]
        Ey = inv_fourier[:, :, None, :] @ Sy[:, :, None, None, :, :]
        Ez = inv_fourier[:, :, None, :] @ Sz[:, :, None, None, :, :]
        Hx = 1j * inv_fourier[:, :, None, :] @ Ux[:, :, None, None, :, :]
        Hy = 1j * inv_fourier[:, :, None, :] @ Uy[:, :, None, None, :, :]
        Hz = 1j * inv_fourier[:, :, None, :] @ Uz[:, :, None, None, :, :]

        val = torch.cat(
            (Ex.squeeze(-1), Ey.squeeze(-1), Ez.squeeze(-1), Hx.squeeze(-1), Hy.squeeze(-1), Hz.squeeze(-1)), -1)
        val = torch.moveaxis(val, 1, 0)

        field_cell[:, res_z * idx_layer:res_z * (idx_layer + 1)] = val

        T_layer = big_A_i @ big_X @ T_layer

    return field_cell


def field_dist_2d(wavelength, kx, ky, T1, layer_info_list, period,
                  res_x=20, res_y=20, res_z=20, set_field_input=(True, False, False),
                  device='cpu', type_complex=torch.complex128):
    k0 = 2 * torch.pi / wavelength

    ff_x = len(kx)
    ff_y = len(ky)
    ff_xy = ff_x * ff_y

    Kx = torch.diag(torch.tile(kx, (ff_y,)).flatten())
    Ky = torch.diag(torch.tile(ky.reshape((-1, 1)), (ff_x,)).flatten())

    # import time
    # t0 = time.time()
    # field_cell0 = torch.zeros((res_z * len(layer_info_list), res_y, res_x, 6), device=device, dtype=type_complex)
    #
    # T_layer = T1[0]
    # big_I = torch.eye((len(T1[0])), device=device, dtype=type_complex)
    #
    # # From the first layer
    # for idx_layer, (epz_conv_i, W, V, q, d, big_A_i, big_B) in enumerate(layer_info_list[::-1]):
    #
    #     W_11 = W[:ff_xy, :ff_xy]
    #     W_12 = W[:ff_xy, ff_xy:]
    #     W_21 = W[ff_xy:, :ff_xy]
    #     W_22 = W[ff_xy:, ff_xy:]
    #
    #     V_11 = V[:ff_xy, :ff_xy]
    #     V_12 = V[:ff_xy, ff_xy:]
    #     V_21 = V[ff_xy:, :ff_xy]
    #     V_22 = V[ff_xy:, ff_xy:]
    #
    #     q_1 = q[:ff_xy]
    #     q_2 = q[ff_xy:]
    #
    #     big_X = torch.diag(torch.exp(-k0 * q * d))
    #
    #     c = torch.cat([big_I, big_B @ big_A_i @ big_X])  @ T_layer
    #
    #     z_1d = torch.linspace(0, res_z, res_z, device=device, dtype=type_complex).reshape((-1, 1, 1)) / res_z * d
    #     # z_1d = np.arange(0, res_z, res_z).reshape((-1, 1, 1)) / res_z * d
    #
    #     c1_plus0 = c[0 * ff_xy:1 * ff_xy]
    #     c2_plus0 = c[1 * ff_xy:2 * ff_xy]
    #     c1_minus0 = c[2 * ff_xy:3 * ff_xy]
    #     c2_minus0 = c[3 * ff_xy:4 * ff_xy]
    #
    #     big_Q1 = torch.diag(q_1)
    #     big_Q2 = torch.diag(q_2)
    #
    #     Sx0 = W_11 @ (d_exp(-k0 * big_Q1 * z_1d) @ c1_plus0 + d_exp(k0 * big_Q1 * (z_1d - d)) @ c1_minus0) \
    #           + W_12 @ (d_exp(-k0 * big_Q2 * z_1d) @ c2_plus0 + d_exp(k0 * big_Q2 * (z_1d - d)) @ c2_minus0)
    #     Sy0 = W_21 @ (d_exp(-k0 * big_Q1 * z_1d) @ c1_plus0 + d_exp(k0 * big_Q1 * (z_1d - d)) @ c1_minus0) \
    #           + W_22 @ (d_exp(-k0 * big_Q2 * z_1d) @ c2_plus0 + d_exp(k0 * big_Q2 * (z_1d - d)) @ c2_minus0)
    #
    #     Ux0 = V_11 @ (-d_exp(-k0 * big_Q1 * z_1d) @ c1_plus0 + d_exp(k0 * big_Q1 * (z_1d - d)) @ c1_minus0) \
    #          + V_12 @ (-d_exp(-k0 * big_Q2 * z_1d) @ c2_plus0 + d_exp(k0 * big_Q2 * (z_1d - d)) @ c2_minus0)
    #     Uy0 = V_21 @ (-d_exp(-k0 * big_Q1 * z_1d) @ c1_plus0 + d_exp(k0 * big_Q1 * (z_1d - d)) @ c1_minus0) \
    #          + V_22 @ (-d_exp(-k0 * big_Q2 * z_1d) @ c2_plus0 + d_exp(k0 * big_Q2 * (z_1d - d)) @ c2_minus0)
    #
    #     Sz0 = -1j * epz_conv_i @ (Kx @ Uy0 - Ky @ Ux0)
    #     Uz0 = -1j * (Kx @ Sy0 - Ky @ Sx0)
    #
    #     # x_1d = np.arange(res_x).reshape((1, -1, 1)) * period[0] / res_x
    #     x_1d = torch.linspace(0, period[0], res_x, device=device, dtype=type_complex).reshape((1, -1, 1))
    #
    #     # y_1d = np.arange(res_y-1, -1, -1).reshape((-1, 1, 1)) * period[1] / res_y
    #     # y_1d = torch.linspace(0, period[1], res_y, device=device, dtype=type_complex)[::-1].reshape((-1, 1, 1))
    #     y_1d = torch.flip(torch.linspace(0, period[1], res_y, device=device, dtype=type_complex), dims=(0,)).reshape((-1, 1, 1))
    #
    #     x_2d = torch.tile(x_1d, (res_y, 1, 1))
    #     x_2d = x_2d * kx * k0
    #     x_2d = x_2d.reshape((res_y, res_x, 1, len(kx)))
    #
    #     y_2d = torch.tile(y_1d, (1, res_x, 1))
    #     y_2d = y_2d * ky * k0
    #     y_2d = y_2d.reshape((res_y, res_x, len(ky), 1))
    #
    #     inv_fourier = torch.exp(-1j * x_2d) * torch.exp(-1j * y_2d)
    #     inv_fourier = inv_fourier.reshape((res_y, res_x, -1))
    #     # (20, 50, 1, 63)   (50, 1, 1, 63, 1);  50 20 50 1 1
    #     Ex0 = inv_fourier[:, :, None, :] @ Sx0[:, None, None, :, :]
    #     Ey0 = inv_fourier[:, :, None, :] @ Sy0[:, None, None, :, :]
    #     Ez0 = inv_fourier[:, :, None, :] @ Sz0[:, None, None, :, :]
    #     Hx0 = 1j * inv_fourier[:, :, None, :] @ Ux0[:, None, None, :, :]
    #     Hy0 = 1j * inv_fourier[:, :, None, :] @ Uy0[:, None, None, :, :]
    #     Hz0 = 1j * inv_fourier[:, :, None, :] @ Uz0[:, None, None, :, :]
    #
    #     val = torch.cat(
    #         (Ex0.squeeze(-1), Ey0.squeeze(-1), Ez0.squeeze(-1), Hx0.squeeze(-1), Hy0.squeeze(-1), Hz0.squeeze(-1)), -1)
    #
    #     field_cell0[res_z * idx_layer:res_z * (idx_layer + 1)] = val
    #
    #     T_layer = big_A_i @ big_X @ T_layer
    #
    # print('original:', time.time() - t0)

    # t0 = time.time()

    # set_field_input = [True, False, False]
    # set_field_input = [True, False, True]
    # set_field_input = [True, True, True]
    field_cell = torch.zeros((sum(set_field_input), res_z * len(layer_info_list), res_y, res_x, 6), device=device,
                             dtype=type_complex)
    T_layer = T1[list(set_field_input)]

    big_I = torch.eye((len(T1[0])), device=device, dtype=type_complex)

    # From the first layer
    for idx_layer, (epz_conv_i, W, V, q, d, big_A_i, big_B) in enumerate(layer_info_list[::-1]):
        W_11 = W[:ff_xy, :ff_xy]
        W_12 = W[:ff_xy, ff_xy:]
        W_21 = W[ff_xy:, :ff_xy]
        W_22 = W[ff_xy:, ff_xy:]

        V_11 = V[:ff_xy, :ff_xy]
        V_12 = V[:ff_xy, ff_xy:]
        V_21 = V[ff_xy:, :ff_xy]
        V_22 = V[ff_xy:, ff_xy:]

        q_1 = q[:ff_xy]
        q_2 = q[ff_xy:]

        big_X = torch.diag(torch.exp(-k0 * q * d))

        c = torch.cat([big_I, big_B @ big_A_i @ big_X]) @ T_layer

        z_1d = torch.linspace(0, res_z, res_z, device=device, dtype=type_complex).reshape((-1, 1, 1)) / res_z * d
        # z_1d = np.arange(0, res_z, res_z).reshape((-1, 1, 1)) / res_z * d

        c1_plus = c[:, 0 * ff_xy:1 * ff_xy]
        c2_plus = c[:, 1 * ff_xy:2 * ff_xy]
        c1_minus = c[:, 2 * ff_xy:3 * ff_xy]
        c2_minus = c[:, 3 * ff_xy:4 * ff_xy]

        big_Q1 = torch.diag(q_1)
        big_Q2 = torch.diag(q_2)

        Sx = W_11 @ (d_exp(-k0 * big_Q1 * z_1d)[:, None, :, :] @ c1_plus
                     + d_exp(k0 * big_Q1 * (z_1d - d))[:, None, :, :] @ c1_minus) \
             + W_12 @ (d_exp(-k0 * big_Q2 * z_1d)[:, None, :, :] @ c2_plus
                       + d_exp(k0 * big_Q2 * (z_1d - d))[:, None, :, :] @ c2_minus)
        Sy = W_21 @ (d_exp(-k0 * big_Q1 * z_1d)[:, None, :, :] @ c1_plus
                     + d_exp(k0 * big_Q1 * (z_1d - d))[:, None, :, :] @ c1_minus) \
             + W_22 @ (d_exp(-k0 * big_Q2 * z_1d)[:, None, :, :] @ c2_plus
                       + d_exp(k0 * big_Q2 * (z_1d - d))[:, None, :, :] @ c2_minus)
        Ux = V_11 @ (-d_exp(-k0 * big_Q1 * z_1d)[:, None, :, :] @ c1_plus
                     + d_exp(k0 * big_Q1 * (z_1d - d))[:, None, :, :] @ c1_minus) \
             + V_12 @ (-d_exp(-k0 * big_Q2 * z_1d)[:, None, :, :] @ c2_plus
                       + d_exp(k0 * big_Q2 * (z_1d - d))[:, None, :, :] @ c2_minus)
        Uy = V_21 @ (-d_exp(-k0 * big_Q1 * z_1d)[:, None, :, :] @ c1_plus
                     + d_exp(k0 * big_Q1 * (z_1d - d))[:, None, :, :] @ c1_minus) \
             + V_22 @ (-d_exp(-k0 * big_Q2 * z_1d)[:, None, :, :] @ c2_plus
                       + d_exp(k0 * big_Q2 * (z_1d - d))[:, None, :, :] @ c2_minus)

        Sz = -1j * epz_conv_i @ (Kx @ Uy - Ky @ Ux)
        Uz = -1j * (Kx @ Sy - Ky @ Sx)

        # x_1d = np.arange(res_x).reshape((1, -1, 1)) * period[0] / res_x
        x_1d = torch.linspace(0, period[0], res_x, device=device, dtype=type_complex).reshape((1, -1, 1))

        # y_1d = np.arange(res_y-1, -1, -1).reshape((-1, 1, 1)) * period[1] / res_y
        # y_1d = torch.linspace(0, period[1], res_y, device=device, dtype=type_complex)[::-1].reshape((-1, 1, 1))
        y_1d = torch.flip(torch.linspace(0, period[1], res_y, device=device, dtype=type_complex), dims=(0,)).reshape(
            (-1, 1, 1))

        x_2d = torch.tile(x_1d, (res_y, 1, 1))
        x_2d = x_2d * kx * k0
        x_2d = x_2d.reshape((res_y, res_x, 1, len(kx)))

        y_2d = torch.tile(y_1d, (1, res_x, 1))
        y_2d = y_2d * ky * k0
        y_2d = y_2d.reshape((res_y, res_x, len(ky), 1))

        inv_fourier = torch.exp(-1j * x_2d) * torch.exp(-1j * y_2d)
        inv_fourier = inv_fourier.reshape((res_y, res_x, -1))

        Ex = inv_fourier[:, :, None, :] @ Sx[:, :, None, None, :, :]
        Ey = inv_fourier[:, :, None, :] @ Sy[:, :, None, None, :, :]
        Ez = inv_fourier[:, :, None, :] @ Sz[:, :, None, None, :, :]
        Hx = 1j * inv_fourier[:, :, None, :] @ Ux[:, :, None, None, :, :]
        Hy = 1j * inv_fourier[:, :, None, :] @ Uy[:, :, None, None, :, :]
        Hz = 1j * inv_fourier[:, :, None, :] @ Uz[:, :, None, None, :, :]

        val = torch.cat(
            (Ex.squeeze(-1), Ey.squeeze(-1), Ez.squeeze(-1), Hx.squeeze(-1), Hy.squeeze(-1), Hz.squeeze(-1)), -1)
        val = torch.moveaxis(val, 1, 0)
        field_cell[:, res_z * idx_layer:res_z * (idx_layer + 1)] = val

        T_layer = big_A_i @ big_X @ T_layer

    # print('new:', time.time() - t0)

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


def d_exp(x):
    res = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
    ix = torch.arange(x.shape[-1], device=x.device)
    res[:, ix, ix] = torch.exp(x[:, ix, ix])
    return res
