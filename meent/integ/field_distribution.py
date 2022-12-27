import meent as ee
import matplotlib.pyplot as plt

from scipy.linalg import expm


def field_distribution(grating_type, *args, **kwargs):
    if grating_type == 0:
        res = field_dist_1d(*args, **kwargs)
    else:
        res = field_dist_2d(*args, **kwargs)
    return res


def field_dist_1d(wavelength, n_I, theta, fourier_order, T1, layer_info_list, period, pol, resolution=(100, 1, 100)):

    k0 = 2 * ee.pi / wavelength
    fourier_indices = ee.arange(-fourier_order, fourier_order + 1)

    kx_vector = k0 * (n_I * ee.sin(theta) - fourier_indices * (wavelength / period[0])).astype('complex')
    Kx = ee.diag(kx_vector / k0)

    resolution_z, resolution_y, resolution_x = resolution

    field_cell = ee.zeros((resolution_z * len(layer_info_list), resolution_y, resolution_x, 3), dtype='complex')

    T_layer = T1

    # From the first layer
    for idx_layer, (E_conv_i, q, W, X, a_i, b, d) in enumerate(layer_info_list[::-1]):

        c1 = T_layer[:, None]
        c2 = b @ a_i @ X @ T_layer[:, None]

        Q = ee.diag(q)

        if pol == 0:
            V = W @ Q

        else:
            V = E_conv_i @ W @ Q
            EKx = E_conv_i @ Kx

        for k in range(resolution_z):
            z = k / resolution_z * d

            if pol == 0:  # TE
                Sy = W @ (expm(-k0 * Q * z) @ c1 + expm(k0 * Q * (z - d)) @ c2)
                Ux = V @ (-expm(-k0 * Q * z) @ c1 + expm(k0 * Q * (z - d)) @ c2)
                f_here = (-1j) * Kx @ Sy

                for j in range(resolution_y):
                    for i in range(resolution_x):
                        x = i * period[0] / resolution_x

                        Ey = Sy.T @ ee.exp(-1j * kx_vector.reshape((-1, 1)) * x)
                        Hx = -1j * Ux.T @ ee.exp(-1j * kx_vector.reshape((-1, 1)) * x)
                        Hz = f_here.T @ ee.exp(-1j * kx_vector.reshape((-1, 1)) * x)

                        field_cell[resolution_z * idx_layer + k, j, i] = Ey, Hx, Hz
            else:  # TM
                Uy = W @ (expm(-k0 * Q * z) @ c1 + expm(k0 * Q * (z - d)) @ c2)
                Sx = V @ (-expm(-k0 * Q * z) @ c1 + expm(k0 * Q * (z - d)) @ c2)

                f_here = (-1j) * EKx @ Uy

                for j in range(resolution_y):
                    for i in range(resolution_x):
                        x = i * period[0] / resolution_x

                        Hy = Uy.T @ ee.exp(-1j * kx_vector.reshape((-1, 1)) * x)
                        Ex = 1j * Sx.T @ ee.exp(-1j * kx_vector.reshape((-1, 1)) * x)
                        Ez = f_here.T @ ee.exp(-1j * kx_vector.reshape((-1, 1)) * x)

                        field_cell[resolution_z * idx_layer + k, j, i] = Hy, Ex, Ez

        T_layer = a_i @ X @ T_layer

    return field_cell


def field_dist_2d(wavelength, n_I, theta, phi, fourier_order, T1, layer_info_list, period, resolution=(100, 100, 100)):
    k0 = 2 * ee.pi / wavelength
    fourier_indices = ee.arange(-fourier_order, fourier_order + 1)
    ff = 2 * fourier_order + 1

    kx_vector = k0 * (n_I * ee.sin(theta) * ee.cos(phi) - fourier_indices * (
            wavelength / period[0])).astype('complex')
    ky_vector = k0 * (n_I * ee.sin(theta) * ee.sin(phi) - fourier_indices * (
            wavelength / period[1])).astype('complex')

    Kx = ee.diag(ee.tile(kx_vector, ff).flatten()) / k0
    Ky = ee.diag(ee.tile(ky_vector.reshape((-1, 1)), ff).flatten()) / k0

    resolution_z, resolution_y, resolution_x = resolution
    field_cell = ee.zeros((resolution_z * len(layer_info_list), resolution_y, resolution_x, 6), dtype='complex')

    T_layer = T1

    big_I = ee.eye((len(T1)))

    # From the first layer
    for idx_layer, (E_conv_i, q, W_11, W_12, W_21, W_22, V_11, V_12, V_21, V_22, big_X, big_A_i, big_B, d)\
            in enumerate(layer_info_list[::-1]):

        c = ee.block([[big_I], [big_B @ big_A_i @ big_X]]) @ T_layer

        ff = len(c) // 4

        c1_plus = c[0*ff:1*ff]
        c2_plus = c[1*ff:2*ff]
        c1_minus = c[2*ff:3*ff]
        c2_minus = c[3*ff:4*ff]

        q1 = q[:len(q)//2]
        q2 = q[len(q)//2:]
        big_Q1 = ee.diag(q1)
        big_Q2 = ee.diag(q2)

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

                    exp_K = ee.exp(-1j * kx_vector.reshape((1, -1)) * x) * ee.exp(-1j * ky_vector.reshape((-1, 1)) * y)
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


def field_plot_zx(field_cell, pol=0, plot_indices=(1, 1, 1, 1, 1, 1), y_slice=0, z_slice=-1, zx=True, yx=True):

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
                plt.clim(0, 3.5)  # identical to caxis([-4,4]) in MATLAB
                plt.colorbar()
                plt.title(title[idx])
                plt.show()

    # for idx in range(len(title)):
    #     if plot_indices[idx]:
    #         plt.imshow((abs(field_cell[0, :, :, idx]) ** 2), cmap='jet', aspect='auto')
    #         # plt.clim(0, 1.3)  # identical to caxis([-4,4]) in MATLAB
    #         plt.colorbar()
    #         plt.title(title[idx])
    #         plt.show()
    # for idx in range(len(title)):
    #     if plot_indices[idx]:
    #         plt.imshow((abs(field_cell[-1, :, :, idx]) ** 2), cmap='jet', aspect='auto')
    #         # plt.clim(0, 3.2)  # identical to caxis([-4,4]) in MATLAB
    #         plt.colorbar()
    #         plt.title(title[idx])
    #         plt.show()
