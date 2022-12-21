import scipy
import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import expm

# def field_dist(f, x, y, z, T, Q, T1):
#
#     X = np.diag(np.exp(-k0 * q * d))
#
#     W_i = np.linalg.inv(W)
#     V_i = np.linalg.inv(V)
#
#     a = 0.5 * (W_i @ f1 + V_i @ g1)
#     b = 0.5 * (W_i @ f1 - V_i @ g1)
#
#     a_i = np.linalg.inv(a)
#
#     c1 = T1[:, None]
#     c2 = b @ a_i @ X @ T1[:, None]
#
#     step = self.period[0] / len_x
#
#     x *= step
#
#     Uy = W @ (scipy.linalg.expm(-k0 * Q * z) @ c1 + scipy.linalg.expm(k0 * Q * (z - d)) @ c2)
#     Hy = Uy.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x)
#
#     # Original TMM method (without enhanced)
#     # Z_II = np.diag(k_II_z / (k0 * self.n_II ** 2))
#     # CC = np.linalg.inv(np.block([[W@X, W], [V@X, -V]])) @ np.block([[np.eye(self.ff)], [1j*Z_II]]) @ T
#     # cc1 = CC[:self.ff]
#     # cc2 = CC[self.ff:]
#     # Uy1 = W @ (scipy.linalg.expm(-k0 * Q * z) @ cc1 + scipy.linalg.expm(k0 * Q * (z - d)) @ cc2)
#     # Hy1 = Uy1.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x)
#
#
#     omega = 2 * np.pi / wl
#     G = (1j * k0 / omega) * W @ Q @ (-scipy.linalg.expm(-k0 * Q * z) @ c1 + scipy.linalg.expm(k0 * Q * (z - d)) @ c2)
#     Dx = G.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x)
#     Ex = Dx if x < 300 else Dx /3.48**2
#
#     # eps0 = 8.854E-12/1E-9
#     eps0 = 100 * np.sqrt(2)
#     eps0= 1/ omega
#     f_here = (-1j / omega / eps0) * np.linalg.inv(E_conv) @ Kx @ Uy
#     Ez = f_here.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x)
#
#     return Hy, Ex, Ez


# def field_dist_loop_2d_original(kx_vector, ky_vector, T1, big_B, big_A_i, big_X, q, E_conv_i, Kx, Ky, d, k0, period, pol,
#                        W_11, W_12, W_21, W_22, V_11, V_12, V_21, V_22):
#
#     field_cell = np.zeros((100, 100, 100, 6), dtype='complex')
#     len_x, len_y, len_z = field_cell.shape[:3]
#
#     big_I = np.eye((len(big_B)))
#     c = np.block([[big_I], [big_B @ big_A_i @ big_X]]) @ T1
#
#     ff = len(c) // 4
#
#     c1_plus = c[0*ff:1*ff]
#     c2_plus = c[1*ff:2*ff]
#     c1_minus = c[2*ff:3*ff]
#     c2_minus = c[3*ff:4*ff]
#
#     q1 = q[:len(q)//2]
#     q2 = q[len(q)//2:]
#     big_Q1 = np.diag(q1)
#     big_Q2 = np.diag(q2)
#
#     for k in range(len_z):
#         z = k / len_z * d
#
#         Sx = W_11 @ (expm(-k0 * big_Q1 * z) @ c1_plus + expm(k0 * big_Q1 * (z-d)) @ c1_minus) \
#              + W_12 @ (expm(-k0 * big_Q2 * z) @ c2_plus + expm(k0 * big_Q2 * (z-d)) @ c2_minus)
#
#         Sy = W_21 @ (expm(-k0 * big_Q1 * z) @ c1_plus + expm(k0 * big_Q1 * (z-d)) @ c1_minus) \
#              + W_22 @ (expm(-k0 * big_Q2 * z) @ c2_plus + expm(k0 * big_Q2 * (z-d)) @ c2_minus)
#
#         Ux = V_11 @ (-expm(-k0 * big_Q1 * z) @ c1_plus + expm(k0 * big_Q1 * (z-d)) @ c1_minus) \
#              + V_12 @ (-expm(-k0 * big_Q2 * z) @ c2_plus + expm(k0 * big_Q2 * (z-d)) @ c2_minus)
#
#         Uy = V_21 @ (-expm(-k0 * big_Q1 * z) @ c1_plus + expm(k0 * big_Q1 * (z-d)) @ c1_minus) \
#              + V_22 @ (-expm(-k0 * big_Q2 * z) @ c2_plus + expm(k0 * big_Q2 * (z-d)) @ c2_minus)
#
#         Sz = -1j * (Kx @ Uy - Ky @ Ux)
#
#         Uz = -1j * (Kx @ Sy - Ky @ Sx)
#
#         for j in range(len_y):
#             y = j * period[1] / len_y
#
#             for i in range(len_x):
#                 x = i * period[0] / len_x
#
#                 exp_K = np.exp(-1j*kx_vector.reshape((1, -1)) * x) * np.exp(-1j*ky_vector.reshape((-1, 1)) * y)
#                 exp_K = exp_K.flatten()
#
#                 Ex = Sx.T @ exp_K
#                 Ey = Sy.T @ exp_K
#                 Ez = Sz.T @ exp_K
#
#                 Hx = Ux.T @ exp_K
#                 Hy = Uy.T @ exp_K
#                 Hz = Uz.T @ exp_K
#
#                 field_cell[i, j, k] = [Ex, Ey, Ez, Hx, Hy, Hz]
#
#     return field_cell


def field_distribution(grating_type, *args, **kwargs):
    if grating_type == 0:
        res = field_dist_loop_1d(*args, **kwargs)
    else:
        res = field_dist_loop_2d(*args, **kwargs)
    return res


def field_dist_loop_2d(kx_vector, ky_vector, T1, layer_info_list, Kx, Ky, k0, period, pol,
                       resolution=(100, 100, 100)):

    resolution_z, resolution_y, resolution_x = resolution
    field_cell = np.zeros((resolution_z * len(layer_info_list), resolution_y, resolution_x, 6), dtype='complex')

    T_layer = T1

    big_I = np.eye((len(T1)))

    # From the first layer
    for idx_layer, (E_conv_i, q, W_11, W_12, W_21, W_22, V_11, V_12, V_21, V_22, big_X, big_A_i, big_B, d)\
            in enumerate(layer_info_list[::-1]):

        c = np.block([[big_I], [big_B @ big_A_i @ big_X]]) @ T_layer

        ff = len(c) // 4

        c1_plus = c[0*ff:1*ff]
        c2_plus = c[1*ff:2*ff]
        c1_minus = c[2*ff:3*ff]
        c2_minus = c[3*ff:4*ff]

        q1 = q[:len(q)//2]
        q2 = q[len(q)//2:]
        big_Q1 = np.diag(q1)
        big_Q2 = np.diag(q2)

        for k in range(resolution_z):
            z = k / resolution_z * d

            Sx = W_11 @ (expm(-k0 * big_Q1 * z) @ c1_plus + expm(k0 * big_Q1 * (z-d)) @ c1_minus) \
                 + W_12 @ (expm(-k0 * big_Q2 * z) @ c2_plus + expm(k0 * big_Q2 * (z-d)) @ c2_minus)

            Sy = W_21 @ (expm(-k0 * big_Q1 * z) @ c1_plus + expm(k0 * big_Q1 * (z-d)) @ c1_minus) \
                 + W_22 @ (expm(-k0 * big_Q2 * z) @ c2_plus + expm(k0 * big_Q2 * (z-d)) @ c2_minus)

            Ux = V_11 @ (-expm(-k0 * big_Q1 * z) @ c1_plus + expm(k0 * big_Q1 * (z-d)) @ c1_minus) \
                 + V_12 @ (-expm(-k0 * big_Q2 * z) @ c2_plus + expm(k0 * big_Q2 * (z-d)) @ c2_minus)

            Uy = V_21 @ (-expm(-k0 * big_Q1 * z) @ c1_plus + expm(k0 * big_Q1 * (z-d)) @ c1_minus) \
                 + V_22 @ (-expm(-k0 * big_Q2 * z) @ c2_plus + expm(k0 * big_Q2 * (z-d)) @ c2_minus)

            Sz = -1j * E_conv_i @ (Kx @ Uy - Ky @ Ux)

            Uz = -1j * (Kx @ Sy - Ky @ Sx)

            for j in range(resolution_y):
                y = j * period[1] / resolution_y

                for i in range(resolution_x):
                    x = i * period[0] / resolution_x

                    exp_K = np.exp(-1j*kx_vector.reshape((1, -1)) * x) * np.exp(-1j*ky_vector.reshape((-1, 1)) * y)
                    exp_K = exp_K.flatten()

                    Ex = Sx.T @ exp_K
                    Ey = Sy.T @ exp_K
                    Ez = Sz.T @ exp_K

                    Hx = -1j * Ux.T @ exp_K
                    Hy = -1j * Uy.T @ exp_K
                    Hz = -1j * Uz.T @ exp_K

                    field_cell[resolution_z * idx_layer + k, j, i] = [Ex, Ey, Ez, Hx, Hy, Hz]

        T_layer = big_A_i @ big_X @ T_layer

    return field_cell


def field_dist_loop_1d(kx_vector, T1, layer_info_list, Kx, k0, period, pol, resolution=(100, 1, 100)):

    resolution_z, resolution_y, resolution_x = resolution

    field_cell = np.zeros((resolution_z * len(layer_info_list), resolution_y, resolution_x, 3), dtype='complex')

    T_layer = T1

    # From the first layer
    for idx_layer, (E_conv_i, Q, W, X, a_i, b, d) in enumerate(layer_info_list[::-1]):

        c1 = T_layer[:, None]
        c2 = b @ a_i @ X @ T_layer[:, None]

        if pol == 0:
            V = W @ Q

        else:
            V = E_conv_i @ W @ Q
            EKx = E_conv_i @ Kx

        for k in range(resolution_z):
            z = k / resolution_z * d

            # S = V @ (-expm(-k0 * Q * z) @ c1 + expm(k0 * Q * (z - d)) @ c2)
            # Ex1 = 1j * S.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x)

            if pol == 0:  # TE

                Sy = W @ (expm(-k0 * Q * z) @ c1 + expm(k0 * Q * (z - d)) @ c2)
                Ux = V @ (-expm(-k0 * Q * z) @ c1 + expm(k0 * Q * (z - d)) @ c2)
                f_here = (-1j) * Kx @ Sy

                for j in range(resolution_y):
                    for i in range(resolution_x):
                        x = i * period[0] / resolution_x
                        Ey = Sy.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x)
                        Hx = -1j * Ux.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x)
                        Hz = f_here.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x)

                        field_cell[i, j, resolution_z*idx_layer + k] = Ey, Hx, Hz
            else:  # TM
                Uy = W @ (expm(-k0 * Q * z) @ c1 + expm(k0 * Q * (z - d)) @ c2)
                Sx = V @ (-expm(-k0 * Q * z) @ c1 + expm(k0 * Q * (z - d)) @ c2)

                f_here = (-1j) * EKx @ Uy

                for j in range(resolution_y):
                    for i in range(resolution_x):
                        x = i * period[0] / resolution_x

                        Hy = Uy.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x)
                        Ex = 1j * Sx.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x)
                        Ez = f_here.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x)

                        field_cell[resolution_z * idx_layer + k, j, i] = Hy, Ex, Ez

        T_layer = a_i @ X @ T_layer

    return field_cell

# def field_dist_loop_1d(kx_vector, T1, layer_info_list, Kx, k0, period, pol):
#
#     field_cell = np.zeros((100, 1, 100*len(layer_info_list), 3), dtype='complex')
#
#     T_layer = T1
#
#     # From the first layer
#     for idx_layer, (E_conv_i, Q, W, X, a_i, b, d) in enumerate(layer_info_list[::-1]):
#
#         len_x, len_y, len_z = field_cell.shape[:3]
#         len_x, len_y, len_z = 100, 1, 100
#
#         c1 = T_layer[:, None]
#         c2 = b @ a_i @ X @ T_layer[:, None]
#
#         WQ = W @ Q
#
#         V = W @ Q if pol == 0 else E_conv_i @ W @ Q
#
#         for k in range(len_z):
#             z = k / len_z * d
#
#             # S = V @ (-expm(-k0 * Q * z) @ c1 + expm(k0 * Q * (z - d)) @ c2)
#             # Ex1 = 1j * S.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x)
#
#             if pol:  # TM
#                 Uy = W @ (expm(-k0 * Q * z) @ c1 + expm(k0 * Q * (z - d)) @ c2)
#                 G = 1j * WQ @ (-expm(-k0 * Q * z) @ c1 + expm(k0 * Q * (z - d)) @ c2)
#                 EKx = E_conv_i @ Kx
#
#                 f_here = (-1j) * EKx @ Uy
#
#                 for j in range(len_y):
#                     for i in range(len_x):
#                         x = i * period[0] / len_x
#
#                         Hy = Uy.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x)
#                         Dx = G.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x)
#                         Ex = Dx if not x < 300 else Dx / 3.48 ** 2  # TODO: make it general
#
#                         Ez = f_here.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x)
#
#                         field_cell[i, j, 100*idx_layer + k] = Hy, Ex, Ez
#             else:  # TE
#
#                 Sy = W @ (expm(-k0 * Q * z) @ c1 + expm(k0 * Q * (z - d)) @ c2)
#                 G = 1j * WQ @ (-expm(-k0 * Q * z) @ c1 + expm(k0 * Q * (z - d)) @ c2)
#                 f_here = (-1j) * Kx @ Sy
#
#                 for j in range(len_y):
#                     for i in range(len_x):
#                         x = i * period[0] / len_x
#                         Ey = Sy.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x)
#                         Hx = G.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x)
#                         Hz = f_here.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x)
#
#                         field_cell[i, j, 100*idx_layer + k] = Ey, Hx, Hz
#
#         T_layer = a_i @ X @ T_layer
#
#     return field_cell


# def field_dist_1d_tm(x, kx_vector, Uy, G, f_here):
#
#     Hy = Uy.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x)
#     Dx = G.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x)
#     Ex = Dx if not x < 300 else Dx / 3.48 ** 2
#     # Ex = Dx if not (x >= 300 and x<= 700 )else Dx / 3.48 **2
#
#     Ez = f_here.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x)
#
#     return Hy, Ex, Ez

# def field_dist_1d_te(x, kx_vector, Sy, G, f_here):
#
#     Ey = Sy.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x)
#     Hx = G.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x)
#     # Ex = Dx if x < 300 else Dx /3.48 **2
#
#     Hz = f_here.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x)
#
#     return Ey, Hx, Hz


# def field_dist_1d_tm(x, kx_vector, Uy, G, f_here):
#
#     # X = np.diag(np.exp(-k0 * q * d))
#     # Q = np.diag(q)
#     #
#     # c1 = T1[:, None]
#     # c2 = b @ a_i @ X @ T1[:, None]
#
#     # Uy = W @ (expm(-k0 * Q * z) @ c1 + expm(k0 * Q * (z - d)) @ c2)
#     Hy = Uy.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x)
#
#     # Original TMM method (without enhanced)
#     # Z_II = np.diag(k_II_z / (k0 * self.n_II ** 2))
#     # CC = np.linalg.inv(np.block([[W@X, W], [V@X, -V]])) @ np.block([[np.eye(self.ff)], [1j*Z_II]]) @ T
#     # cc1 = CC[:self.ff]
#     # cc2 = CC[self.ff:]
#     # Uy1 = W @ (scipy.linalg.expm(-k0 * Q * z) @ cc1 + scipy.linalg.expm(k0 * Q * (z - d)) @ cc2)
#     # Hy1 = Uy1.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x)
#
#     # omega = 2 * np.pi / wavelength
#     # G = 1j * W @ Q @ (-expm(-k0 * Q * z) @ c1 + expm(k0 * Q * (z - d)) @ c2)
#     Dx = G.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x)
#     Ex = Dx if x < 300 else Dx /3.48 **2
#
#     # S = V @ (-expm(-k0 * Q * z) @ c1 + expm(k0 * Q * (z - d)) @ c2)
#     # Ex1 = 1j * S.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x)
#
#     # eps0 = 8.854E-12/1E-9
#     # eps0 = 1 / omega
#     # f_here = (-1j) * E_conv_i @ Kx @ Uy
#     Ez = f_here.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x)
#
#     return Hy, Ex, Ez

# def field_dist_1d_tm_vectorize(x_array, y, z_array, T1, q, k0, d, W, b, a_i, kx_vector, wavelength, E_conv_i, Kx):
#
#     z_array = z_array.reshape((-1, 1))
#
#     X = np.diag(np.exp(-k0 * q * d))
#     Q = np.diag(q)
#
#     c1 = T1[:, None]
#     c2 = b @ a_i @ X @ T1[:, None]
#
#
#     Q_hat = np.tile(Q, (50, 1))
#     Q_hat = scipy.linalg.block_diag(*[Q] * 50)
#
#     z_array_hat = np.diag(np.repeat(z_array, 21))
#     z_array_hat = np.vstack(np.repeat(z_array, 21))
#     z_array_hat = (np.repeat(z_array, 21))[:, None]
#     aaa = np.tile(z_array_hat, (1, 21))
#
#     aaaa = Q_hat @ aaa
#     Uy = W @ (expm(-k0 * aaaa) @ c1 + expm(k0 * Q_hat @ (z_array_hat - d)) @ c2)
#     Uy = W @ (expm(-k0 * Q_hat @ z_array_hat) @ c1 + expm(k0 * Q_hat @ (z_array_hat - d)) @ c2)
#
#     c1_hat = np.eye(21) * z_array
#
#
#     Uy = W @ (expm(-k0 * Q * z_array) @ c1 + expm(k0 * Q * (z_array - d)) @ c2)
#     Hy = Uy.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x_array)
#
#     # Original TMM method (without enhanced)
#     # Z_II = np.diag(k_II_z / (k0 * self.n_II ** 2))
#     # CC = np.linalg.inv(np.block([[W@X, W], [V@X, -V]])) @ np.block([[np.eye(self.ff)], [1j*Z_II]]) @ T
#     # cc1 = CC[:self.ff]
#     # cc2 = CC[self.ff:]
#     # Uy1 = W @ (scipy.linalg.expm(-k0 * Q * z) @ cc1 + scipy.linalg.expm(k0 * Q * (z - d)) @ cc2)
#     # Hy1 = Uy1.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x)
#
#     omega = 2 * np.pi / wavelength
#     G = (1j * k0 / omega) * W @ Q @ (-expm(-k0 * Q * z_array) @ c1 + expm(k0 * Q * (z_array - d)) @ c2)
#     Dx = G.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x_array)
#     Ex = Dx if x_array < 300 else Dx /3.48 **2
#
#     # eps0 = 8.854E-12/1E-9
#     eps0 = 1 / omega
#     f_here = (-1j / omega / eps0) * E_conv_i @ Kx @ Uy
#     Ez = f_here.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x_array)
#
#     return Hy, Ex, Ez
def field_plot_zx(field_cell, plot_indices=(1, 1, 1, 1, 1, 1), pol=0):

    if field_cell.shape[-1] == 6:  # 2D grating
        title = ['2D Ex', '2D Ey', '2D Ez', '2D Hx', '2D Hy', '2D Hz', ]
    else:  # 1D grating
        if pol == 0:  # TE
            title = ['1D Ey', '1D Hx', '1D Hz', ]
        else:  # TM
            title = ['1D Hy', '1D Ex', '1D Ez', ]

    for idx in range(len(title)):
        if plot_indices[idx]:
            plt.imshow((abs(field_cell[:, 0, :, idx]) ** 2), cmap='jet', aspect='auto')
            # plt.clim(0, 2)  # identical to caxis([-4,4]) in MATLAB
            plt.colorbar()
            plt.title(title[idx])
            plt.show()
