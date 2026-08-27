"""Reconstructing the field inside the stack, once the solve has finished.

The solve returns amplitudes per order; these routines turn them back into E and H on a grid.
Two steps per layer, in this order:

  1. Propagate. `T_layer` enters at the layer's top face; the mode amplitudes at depth z are
     the entering ones scaled by exp(-k0 q z) for the downward modes and exp(k0 q (z - d)) for
     the upward ones. Both exponents are negative over 0 <= z <= d, which is the point of
     writing the second one against the far face rather than the near one - neither term can
     overflow, however thick the layer or however evanescent the mode.

  2. Invert the Fourier sum. Multiply by exp(-i kx x) and add up the orders.

One routine per grating type, matching the three in `_base`. They iterate `layer_info_list`
reversed because `_base` filled it from the substrate up.

The z axis these produce runs top-down, in the solver's own sweep order; `calculate_field` in
rcwa.py flips it to run the way z does before returning.
"""

import torch


def field_dist_1d(wavelength, kx, T1, layer_info_list, period,
                  pol, res_x=20, res_y=20, res_z=20, device='cpu',
                  type_complex=torch.complex128, type_float=torch.float64):
    """Scalar 1D. Three non-zero components, returned as (Ey, Hx, Hz) for TE, (Hy, Ex, Ez) for TM."""
    k0 = 2 * torch.pi / wavelength
    Kx = torch.diag(kx)

    field_cell = torch.zeros((res_z * len(layer_info_list), res_y, res_x, 3), device=device, dtype=type_complex)

    T_layer = T1

    for idx_layer, (epz_conv_i, W, V, q, d, A_i, B) in enumerate(layer_info_list[::-1]):

        # c1 are the modes travelling away from the entering face, c2 those coming back after
        # the rest of the stack has reflected them.
        X = torch.diag(torch.exp(-k0 * q * d))
        c1 = T_layer[:, None]
        c2 = B @ A_i @ X @ T_layer[:, None]
        Q = torch.diag(q)

        z_1d = torch.linspace(0, res_z, res_z, device=device, dtype=type_complex).reshape((-1, 1, 1)) / res_z * d

        # Note the two exponents: -z for c1, (z - d) for c2. Each is measured from the face its
        # modes enter through, so both stay <= 0 across the layer and neither can overflow.
        # W gives the tangential field the eigenproblem was posed in; V gives its partner,
        # which is why Mx and My differ only by the sign on the c1 term.
        My = W @ (d_exp(-k0 * Q * z_1d) @ c1 + d_exp(k0 * Q * (z_1d - d)) @ c2)
        Mx = V @ (-d_exp(-k0 * Q * z_1d) @ c1 + d_exp(k0 * Q * (z_1d - d)) @ c2)
        # z is not solved for; it follows from the curl equation. TM picks up epz_conv_i
        # because there the z component is an E field and has to be divided by the
        # permittivity - the inverse convolution matrix the layer already carries.
        if pol == 0:
            Mz = -1j * Kx @ My
        else:
            Mz = -1j * epz_conv_i @ Kx @ My if pol else -1j * Kx @ My


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

        val = torch.cat((Fy.squeeze(-1), Fx.squeeze(-1), Fz.squeeze(-1)), -1)

        field_cell[res_z * idx_layer:res_z * (idx_layer + 1)] = val

        # Carry the amplitudes across to the next layer's entering face. This is the same
        # product the solve built the stack with, replayed one layer at a time.
        T_layer = A_i @ X @ T_layer

    return field_cell


def field_dist_1d_conical(wavelength, kx, ky, T1, layer_info_list, period,
                          res_x=20, res_y=20, res_z=20, set_field_input=(True, False, False),
                          device='cpu', type_complex=torch.complex128, type_float=torch.float64):
    """1D structure, out-of-plane incidence. All six components.

    `set_field_input` selects which of the three solved incidences to reconstruct - (psi, TE,
    TM), in that order - and its sum becomes the leading axis of the result. Reconstruction is
    much more expensive than the solve, so asking only for what will be looked at is worth it.
    """
    k0 = 2 * torch.pi / wavelength

    ff_x = len(kx)
    ff_y = len(ky)
    ff_xy = ff_x * ff_y

    Kx = torch.diag(torch.tile(kx, (ff_y,)).flatten())
    Ky = torch.diag(torch.tile(ky.reshape((-1, 1)), (ff_x,)).flatten())

    field_cell = torch.zeros((sum(set_field_input), res_z * len(layer_info_list), res_y, res_x, 6), device=device,
                             dtype=type_complex)

    T_layer = T1[list(set_field_input)]

    big_I = torch.eye((len(T1[0])), device=device, dtype=type_complex)
    O = torch.zeros((ff_xy, ff_xy), device=device, dtype=type_complex)

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

        z_1d = torch.linspace(0, res_z, res_z, device=device, dtype=type_complex).reshape((-1, 1, 1)) / res_z * d

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

        x_1d = torch.linspace(0, period[0], res_x, device=device, dtype=type_complex).reshape((1, -1, 1))
        x_2d = torch.tile(x_1d, (res_y, 1, 1))
        x_2d = x_2d * kx * k0
        x_2d = x_2d.reshape((res_y, res_x, 1, len(kx)))

        # y increases with the row index, the same way the ucell's y index does, so a field
        # sample and the ucell cell at the same index sit at the same place. This used to be
        # a flipped grid, which put the two arrays' y axes back to back.
        y_1d = torch.linspace(0, period[1], res_y, device=device, dtype=type_complex).reshape(
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
    """The general case. Same structure as the conical routine, with orders retained in y too.

    The inverse Fourier sum now runs over (x, y) order pairs, so a field map costs
    ff_x * ff_y * res_x * res_y * res_z. Reduce the resolutions before the orders - the orders
    also set the accuracy of the solve, and the resolutions do not.
    """
    k0 = 2 * torch.pi / wavelength

    ff_x = len(kx)
    ff_y = len(ky)
    ff_xy = ff_x * ff_y

    Kx = torch.diag(torch.tile(kx, (ff_y,)).flatten())
    Ky = torch.diag(torch.tile(ky.reshape((-1, 1)), (ff_x,)).flatten())


    field_cell = torch.zeros((sum(set_field_input), res_z * len(layer_info_list), res_y, res_x, 6), device=device,
                             dtype=type_complex)
    T_layer = T1[list(set_field_input)]

    big_I = torch.eye((len(T1[0])), device=device, dtype=type_complex)

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

        x_1d = torch.linspace(0, period[0], res_x, device=device, dtype=type_complex).reshape((1, -1, 1))

        # y increases with the row index, the same way the ucell's y index does, so a field
        # sample and the ucell cell at the same index sit at the same place. This used to be
        # a flipped grid, which put the two arrays' y axes back to back.
        y_1d = torch.linspace(0, period[1], res_y, device=device, dtype=type_complex).reshape(
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


    return field_cell


def field_plot(field_cell, pol=0, plot_indices=(1, 1, 1, 1, 1, 1), y_slice=0, z_slice=-1, zx=True, yx=True):
    """Quick look at a field map. Plots |F|^2, not the complex field.

    A convenience for inspection, not a figure-making tool - it opens one window per component
    and squares away the phase. Anything that needs the sign or the phase should read
    `field_cell` directly.

    matplotlib is imported here rather than at module scope so it stays an optional dependency:
    the solver runs without it and only plotting requires it.
    """
    try:
        import matplotlib.pyplot as plt
    except (ImportError, ModuleNotFoundError) as e:
        print(e)
        print('To use cal_field(), please install matplotlib')
        raise e

    # A 3-component array only reaches here from the scalar 1D path, where which three
    # components those are depends on the polarization.
    if field_cell.shape[-1] == 6:
        title = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz', ]
    else:
        if pol == 0:
            title = ['1D Ey', '1D Hx', '1D Hz', ]
        else:
            title = ['1D Hy', '1D Ex', '1D Ez', ]

    if zx:
        for idx in range(len(title)):
            if plot_indices[idx]:
                plt.imshow((abs(field_cell[:, y_slice, :, idx]) ** 2), cmap='jet', aspect='auto')
                plt.colorbar()
                plt.title(f'{title[idx]}, Side View')
                plt.xlabel('X')
                plt.ylabel('Z')
                plt.show()
    if yx:
        for idx in range(len(title)):
            if plot_indices[idx]:
                plt.imshow((abs(field_cell[z_slice, :, :, idx]) ** 2), cmap='jet', aspect='auto')
                plt.colorbar()
                plt.title(f'{title[idx]}, Top View')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.show()


def d_exp(x):
    """Exponentiate a batch of diagonal matrices, on the diagonal only.

    `torch.exp` applied to the whole matrix would fill every off-diagonal zero with exp(0) = 1,
    which is not the matrix exponential of a diagonal matrix - it is not even close. Only the
    diagonal is touched here and the rest stays zero, which for a diagonal matrix is exactly
    the matrix exponential.
    """
    res = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
    ix = torch.arange(x.shape[-1], device=x.device)
    res[:, ix, ix] = torch.exp(x[:, ix, ix])
    return res
