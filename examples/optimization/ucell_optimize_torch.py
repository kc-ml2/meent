import numpy as np
import jax
import jax.numpy as jnp
import time
import torch

from meent.on_torch.convolution_matrix import put_permittivity_in_ucell, to_conv_mat

from meent.rcwa import call_solver


class RCWAOptimizer:

    def __init__(self, gt, model):
        self.gt = gt
        self.model = model
        pass


if __name__ == '__main__':
    t0 = time.time()

    grating_type = 2

    # Ground Truth
    pol = 1  # 0: TE, 1: TM

    n_I = 1  # n_incidence
    n_II = 1  # n_transmission

    theta = 0
    phi = 0
    psi = 0 if pol else 90

    wavelength = 900

    thickness_gt = [1120]
    ucell_materials = [1, 3.48]
    period = [1000, 1000]
    fourier_order = 4

    mode_key = 2
    device = 0
    dtype = 0

    ucell_gt = torch.tensor(
        [[
            [1., 1., 1., 1., 1., 1., 1., 1., 3, 3.],
            [1., 1., 1., 1., 1., 1., 1., 1., 3, 3.],
            [1., 1., 1., 1., 1., 1., 1., 1., 3, 3.],
            [1., 1., 1., 1., 1., 1., 1., 1., 3, 3.],
            [1., 1., 1., 1., 1., 1., 1., 1., 3, 3.],
            [1., 1., 1., 1., 1., 1., 1., 1., 3, 3.],
            [1., 1., 1., 1., 1., 1., 1., 1., 3, 3.],
            [1., 1., 1., 1., 1., 1., 1., 1., 3, 3.],
            [1., 1., 1., 1., 1., 1., 1., 1., 3, 3.],
            [1., 1., 1., 1., 1., 1., 1., 1., 3, 3.],
        ]], dtype=torch.float64)

    ucell = torch.tensor(
        [[
            [3., 3, 1., 1., 1., 1., 1., 1., 3, 3.],
            [3., 3, 1., 1., 1., 1., 1., 1., 3, 3.],
            [3., 3, 1., 1., 1., 1., 1., 1., 3, 3.],
            [3., 3, 1., 1., 1., 1., 1., 1., 3, 3.],
            [3., 3, 1., 1., 1., 1., 1., 1., 3, 3.],
            [3., 3, 1., 1., 1., 1., 1., 1., 3, 3.],
            [3., 3, 1., 1., 1., 1., 1., 1., 3, 3.],
            [3., 3, 1., 1., 1., 1., 1., 1., 3, 3.],
            [3., 3, 1., 1., 1., 1., 1., 1., 3, 3.],
            [3., 3, 1., 1., 1., 1., 1., 1., 3, 3.],
        ]], dtype=torch.float64, requires_grad=True)

    if mode_key == 0:
        device = None

        if dtype == 0:
            type_complex = np.complex128
        else:
            type_complex = np.complex64

    elif mode_key == 1:
        # JAX
        if device == 0:
            jax.config.update('jax_platform_name', 'cpu')
        else:
            jax.config.update('jax_platform_name', 'gpu')

        if dtype == 0:
            from jax.config import config

            config.update("jax_enable_x64", True)
            type_complex = jnp.complex128
        else:
            type_complex = jnp.complex64

    else:
        # Torch
        if device == 0:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')

        if dtype == 0:
            type_complex = torch.complex128
        else:
            type_complex = torch.complex64

    solver = call_solver(mode_key, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi,
                         psi=psi,
                         fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell,
                         ucell_materials=ucell_materials,
                         thickness=thickness_gt, device=device, type_complex=type_complex, )

    E_conv_all_gt = to_conv_mat(ucell_gt, fourier_order, type_complex=type_complex)
    o_E_conv_all_gt = to_conv_mat(1 / ucell_gt, fourier_order, type_complex=type_complex)

    de_ri_gt, de_ti_gt = solver.solve(wavelength, E_conv_all_gt, o_E_conv_all_gt)

    E_conv_all = to_conv_mat(ucell, fourier_order, type_complex=type_complex)
    o_E_conv_all = to_conv_mat(1 / ucell, fourier_order, type_complex=type_complex)

    de_ri, de_ti = solver.solve(wavelength, E_conv_all, o_E_conv_all)

    opt = torch.optim.Adam([ucell], lr=1E-2)
    for i in range(500):
        E_conv_all = to_conv_mat(ucell, fourier_order, type_complex=type_complex)
        o_E_conv_all = to_conv_mat(1 / ucell, fourier_order, type_complex=type_complex)
        de_ri, de_ti = solver.solve(wavelength, E_conv_all, o_E_conv_all)
        # loss = (torch.linalg.norm(de_ti - de_ti_gt) + torch.linalg.norm(de_ri-de_ri_gt)).sum()
        loss = -de_ti[4, 5]
        loss.backward()
        opt.step()
        opt.zero_grad()
        print(loss)

    print(ucell)
