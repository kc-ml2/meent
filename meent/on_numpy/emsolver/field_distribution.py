import numpy as np


def field_dist_1d(wavelength, kx, T1, layer_info_list, period,
                  pol, res_x=20, res_y=1, res_z=20, type_complex=np.complex128):

    k0 = 2 * np.pi / wavelength
    Kx = np.diag(kx)

    field_cell = np.zeros((res_z * len(layer_info_list), res_y, res_x, 3), dtype=type_complex)

    T_layer = T1

    # From the first layer
    for idx_layer, (epz_conv_i, W, V, q, d, A_i, B) in enumerate(layer_info_list[::-1]):

        X = np.diag(np.exp(-k0 * q * d))
        c1 = T_layer[:, None]
        c2 = B @ A_i @ X @ T_layer[:, None]
        Q = np.diag(q)

        z_1d = np.linspace(0, res_z, res_z).reshape((-1, 1, 1)) / res_z * d

        My = W @ (diag_exp_batch(-k0 * Q * z_1d) @ c1 + diag_exp_batch(k0 * Q * (z_1d - d)) @ c2)
        Mx = V @ (-diag_exp_batch(-k0 * Q * z_1d) @ c1 + diag_exp_batch(k0 * Q * (z_1d - d)) @ c2)

        if pol == 0:
            Mz = -1j * Kx @ My
        else:
            Mz = -1j * epz_conv_i @ Kx @ My if pol else -1j * Kx @ My

        # x_1d = np.arange(1, res_x+1).reshape((1, -1, 1))
        # x_1d = x_1d * period[0] / res_x

        x_1d = np.linspace(0, period[0], res_x).reshape((1, -1, 1))

        x_2d = np.tile(x_1d, (res_y, 1, 1))
        x_2d = x_2d * kx * k0
        x_2d = x_2d.reshape((res_y, res_x, 1, len(kx)))

        inv_fourier = np.exp(-1j * x_2d)
        inv_fourier = inv_fourier.reshape((res_y, res_x, -1))

        if pol == 0:
            Fy = inv_fourier[:, :, None, :] @ My[:, None, None, :, :]
            Fx = 1j * inv_fourier[:, :, None, :] @ Mx[:, None, None, :, :]
            Fz = 1j * inv_fourier[:, :, None, :] @ Mz[:, None, None, :, :]

        else:
            Fy = inv_fourier[:, :, None, :] @ My[:, None, None, :, :]
            Fx = -1j * inv_fourier[:, :, None, :] @ Mx[:, None, None, :, :]
            Fz = -1j * inv_fourier[:, :, None, :] @ Mz[:, None, None, :, :]

        val = np.concatenate((Fy.squeeze(-1), Fx.squeeze(-1), Fz.squeeze(-1)), axis=-1)
        field_cell[res_z * idx_layer:res_z * (idx_layer + 1)] = val

        T_layer = A_i @ X @ T_layer

    return field_cell


def field_dist_2d(wavelength, kx, ky, T1, layer_info_list, period,
                  res_x=20, res_y=20, res_z=20, type_complex=np.complex128):

    k0 = 2 * np.pi / wavelength

    ff_x = len(kx)
    ff_y = len(ky)
    ff_xy = ff_x * ff_y

    Kx = np.diag(np.tile(kx, ff_y).flatten())
    Ky = np.diag(np.tile(ky.reshape((-1, 1)), ff_x).flatten())

    field_cell = np.zeros((res_z * len(layer_info_list), res_y, res_x, 6), dtype=type_complex)

    T_layer = T1

    big_I = np.eye((len(T1))).astype(type_complex)

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

        big_X = np.diag(np.exp(-k0 * q * d))

        c = np.block([[big_I], [big_B @ big_A_i @ big_X]]) @ T_layer
        z_1d = np.linspace(0, res_z, res_z).reshape((-1, 1, 1)) / res_z * d
        # z_1d = np.arange(0, res_z, res_z).reshape((-1, 1, 1)) / res_z * d

        c1_plus = c[0 * ff_xy:1 * ff_xy]
        c2_plus = c[1 * ff_xy:2 * ff_xy]
        c1_minus = c[2 * ff_xy:3 * ff_xy]
        c2_minus = c[3 * ff_xy:4 * ff_xy]

        q1 = q[:len(q) // 2]
        q2 = q[len(q) // 2:]
        big_Q1 = np.diag(q1)
        big_Q2 = np.diag(q2)

        Sx = W_11 @ (diag_exp_batch(-k0 * big_Q1 * z_1d) @ c1_plus + diag_exp_batch(k0 * big_Q1 * (z_1d - d)) @ c1_minus) \
              + W_12 @ (diag_exp_batch(-k0 * big_Q2 * z_1d) @ c2_plus + diag_exp_batch(k0 * big_Q2 * (z_1d - d)) @ c2_minus)
        Sy = W_21 @ (diag_exp_batch(-k0 * big_Q1 * z_1d) @ c1_plus + diag_exp_batch(k0 * big_Q1 * (z_1d - d)) @ c1_minus) \
              + W_22 @ (diag_exp_batch(-k0 * big_Q2 * z_1d) @ c2_plus + diag_exp_batch(k0 * big_Q2 * (z_1d - d)) @ c2_minus)

        # Ux = -V_11 @ (diag_exp_batch(-k0 * big_Q1 * z_1d) @ c1_plus + diag_exp_batch(k0 * big_Q1 * (z_1d - d)) @ c1_minus) \
        #       - V_12 @ (diag_exp_batch(-k0 * big_Q2 * z_1d) @ c2_plus + diag_exp_batch(k0 * big_Q2 * (z_1d - d)) @ c2_minus)
        # Uy = -V_21 @ (diag_exp_batch(-k0 * big_Q1 * z_1d) @ c1_plus + diag_exp_batch(k0 * big_Q1 * (z_1d - d)) @ c1_minus) \
        #       - V_22 @ (diag_exp_batch(-k0 * big_Q2 * z_1d) @ c2_plus + diag_exp_batch(k0 * big_Q2 * (z_1d - d)) @ c2_minus)

        Ux = V_11 @ (-diag_exp_batch(-k0 * big_Q1 * z_1d) @ c1_plus + diag_exp_batch(k0 * big_Q1 * (z_1d - d)) @ c1_minus) \
              + V_12 @ (-diag_exp_batch(-k0 * big_Q2 * z_1d) @ c2_plus + diag_exp_batch(k0 * big_Q2 * (z_1d - d)) @ c2_minus)
        Uy = V_21 @ (-diag_exp_batch(-k0 * big_Q1 * z_1d) @ c1_plus + diag_exp_batch(k0 * big_Q1 * (z_1d - d)) @ c1_minus) \
              + V_22 @ (-diag_exp_batch(-k0 * big_Q2 * z_1d) @ c2_plus + diag_exp_batch(k0 * big_Q2 * (z_1d - d)) @ c2_minus)

        Sz = -1j * epz_conv_i @ (Kx @ Uy - Ky @ Ux)
        Uz = -1j * (Kx @ Sy - Ky @ Sx)

        # x_1d = np.arange(res_x).reshape((1, -1, 1)) * period[0] / res_x
        x_1d = np.linspace(0, period[0], res_x).reshape((1, -1, 1))

        # y_1d = np.arange(res_y-1, -1, -1).reshape((-1, 1, 1)) * period[1] / res_y
        y_1d = np.linspace(0, period[1], res_y)[::-1].reshape((-1, 1, 1))

        x_2d = np.tile(x_1d, (res_y, 1, 1))
        x_2d = x_2d * kx * k0
        x_2d = x_2d.reshape((res_y, res_x, 1, len(kx)))

        y_2d = np.tile(y_1d, (1, res_x, 1))
        y_2d = y_2d * ky * k0
        y_2d = y_2d.reshape((res_y, res_x, len(ky), 1))

        inv_fourier = np.exp(-1j * x_2d) * np.exp(-1j * y_2d)
        inv_fourier = inv_fourier.reshape((res_y, res_x, -1))

        Ex = inv_fourier[:, :, None, :] @ Sx[:, None, None, :, :]
        Ey = inv_fourier[:, :, None, :] @ Sy[:, None, None, :, :]
        Ez = inv_fourier[:, :, None, :] @ Sz[:, None, None, :, :]
        Hx = 1j * inv_fourier[:, :, None, :] @ Ux[:, None, None, :, :]
        Hy = 1j * inv_fourier[:, :, None, :] @ Uy[:, None, None, :, :]
        Hz = 1j * inv_fourier[:, :, None, :] @ Uz[:, None, None, :, :]

        val = np.concatenate(
            (Ex.squeeze(-1), Ey.squeeze(-1), Ez.squeeze(-1), Hx.squeeze(-1), Hy.squeeze(-1), Hz.squeeze(-1)), -1)

        field_cell[res_z * idx_layer:res_z * (idx_layer + 1)] = val

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
    return np.diag(np.exp(np.diag(x)))


def diag_exp_batch(x):
    res = np.zeros(x.shape).astype(x.dtype)
    ix = np.arange(x.shape[-1])
    res[:, ix, ix] = np.exp(x[:, ix, ix])
    return res
