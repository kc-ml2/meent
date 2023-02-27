try:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
except:
    pass

import numpy as np
import jax
import jax.numpy as jnp
import time
from copy import deepcopy

from meent.entrance import call_solver
import torch


def load_setting(mode_key, dtype, device):
    grating_type = 2

    pol = 1  # 0: TE, 1: TM

    n_I = 1  # n_incidence
    n_II = 1  # n_transmission

    theta = 0 * np.pi / 180
    phi = 0 * np.pi / 180
    psi = 0 * np.pi / 180 if pol else 90 * np.pi / 180

    wavelength = 900

    ucell_materials = [1, 3.48]
    fourier_order = 2

    period = [1000, 1000]
    thickness = [1120., 400, 300]
    ucell = np.array(
        [
            [
                [3.1, 1.1, 1.2, 1.6, 3.1],
                [3.5, 1.4, 1.1, 1.2, 3.6],
            ],
            [
                [3.5, 1.2, 1.5, 1.2, 3.3],
                [3.1, 1.5, 1.5, 1.4, 3.1],
            ],
            [
                [3.5, 1.2, 1.5, 1.2, 3.3],
                [3.1, 1.5, 1.5, 1.4, 3.1],
            ],
        ]
    )

    # thickness, period = [1120.], [1000, 1000]
    #
    # ucell = np.array(
    #     [
    #         [
    #             [3.5, 1.2, 1.5, 1.2, 3.3],
    #             [3.1, 1.5, 1.5, 1.4, 3.1],
    #         ],
    #     ]
    # )

    if mode_key == 0:
        device = 0
        type_complex = np.complex128 if dtype == 0 else np.complex64
        ucell = ucell.astype(type_complex)

    elif mode_key == 1:  # JAX
        jax.config.update('jax_platform_name', 'cpu') if device == 0 else jax.config.update('jax_platform_name', 'gpu')

        thickness = jnp.array(thickness)
        period = jnp.array(period)

        if dtype == 0:
            jax.config.update("jax_enable_x64", True)
            type_complex = jnp.complex128
            ucell = ucell.astype(jnp.float64)
            ucell = jnp.array(ucell, dtype=jnp.float64)

        else:
            type_complex = jnp.complex64
            ucell = ucell.astype(jnp.float32)
            ucell = jnp.array(ucell, dtype=jnp.float32)

    else:  # Torch
        device = torch.device('cpu') if device == 0 else torch.device('cuda')
        type_complex = torch.complex128 if dtype == 0 else torch.complex64

        thickness = torch.tensor(thickness)
        period = torch.tensor(period)

        if dtype == 0:
            ucell = torch.tensor(ucell, dtype=torch.float64, device=device)
        else:
            ucell = torch.tensor(ucell, dtype=torch.float32, device=device)

    return grating_type, pol, n_I, n_II, theta, phi, psi, wavelength, thickness, ucell_materials, period, fourier_order,\
           type_complex, device, ucell


def optimize_jax_ucell_metasurface(mode_key, dtype, device):
    from meent.on_jax.emsolver.convolution_matrix import to_conv_mat_discrete

    grating_type, pol, n_I, n_II, theta, phi, psi, wavelength, thickness, ucell_materials, period, fourier_order, \
    type_complex, device, ucell = load_setting(mode_key, dtype, device)


    solver = call_solver(mode_key, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi,
                         psi=psi, fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell,
                         ucell_materials=ucell_materials, thickness=thickness, device=device,
                         type_complex=type_complex, )

    @jax.grad
    def grad_loss(ucell):

        E_conv_all = to_conv_mat_discrete(ucell, fourier_order, type_complex=type_complex)
        o_E_conv_all = to_conv_mat_discrete(1 / ucell, fourier_order, type_complex=type_complex)
        de_ri, de_ti, _, _, _ = solver.solve(wavelength, E_conv_all, o_E_conv_all)
        c = de_ti.shape[0] // 2
        loss = de_ti[c, c+1]
        print(loss.primal)
        return loss

    def grad_numerical(ucell, delta):

        grad_arr = np.zeros(ucell.shape)
        for layer in range(ucell.shape[0]):
            for r in range(ucell.shape[1]):
                for c in range(ucell.shape[2]):
                    ucell_delta_m = ucell.at[layer, r, c].set(ucell[layer, r, c] - delta)

                    E_conv_all_m = to_conv_mat_discrete(ucell_delta_m, fourier_order, type_complex=type_complex)
                    o_E_conv_all_m = to_conv_mat_discrete(1 / ucell_delta_m, fourier_order, type_complex=type_complex)
                    de_ri_delta_m, de_ti_delta_m, _, _, _ = solver.solve(wavelength, E_conv_all_m, o_E_conv_all_m)

                    ucell_delta_p = ucell.at[layer, r, c].set(ucell[layer, r, c] + delta)

                    E_conv_all_p = to_conv_mat_discrete(ucell_delta_p, fourier_order, type_complex=type_complex)
                    o_E_conv_all_p = to_conv_mat_discrete(1 / ucell_delta_p, fourier_order, type_complex=type_complex)
                    de_ri_delta_p, de_ti_delta_p, _, _, _ = solver.solve(wavelength, E_conv_all_p, o_E_conv_all_p)

                    center = de_ti_delta_m.shape[0] // 2
                    grad_numeric = (de_ti_delta_p[center, center+1] - de_ti_delta_m[center, center+1]) / (2*delta)
                    grad_arr[layer, r, c] = grad_numeric

        return grad_arr

    print('JAX grad_ad:\n', grad_loss(ucell))
    print('JAX grad_numeric:\n', grad_numerical(ucell, 1E-6))


def optimize_torch_metasurface(mode_key, dtype, device):
    """
    out of date.
    Will be updated.
    """
    from meent.on_torch.emsolver.convolution_matrix import to_conv_mat_discrete

    grating_type, pol, n_I, n_II, theta, phi, psi, wavelength, thickness, ucell_materials, period, fourier_order, \
    type_complex, device, ucell = load_setting(mode_key, dtype, device)

    ucell.requires_grad = True

    solver = call_solver(mode_key, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi,
                         psi=psi, fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell,
                         ucell_materials=ucell_materials, thickness=thickness, device=device,
                         type_complex=type_complex, )

    E_conv_all = to_conv_mat_discrete(ucell, fourier_order)
    o_E_conv_all = to_conv_mat_discrete(1 / ucell, fourier_order)
    de_ri, de_ti, _, _, _ = solver.solve(wavelength, E_conv_all, o_E_conv_all)

    c = de_ti.shape[0] // 2
    loss = de_ti[c, c + 1]
    loss.backward()
    print('Torch grad_ad:\n', ucell.grad)

    def grad_numerical(ucell, delta):
        ucell.requires_grad = False
        grad_arr = torch.zeros(ucell.shape)

        for layer in range(ucell.shape[0]):
            for r in range(ucell.shape[1]):
                for c in range(ucell.shape[2]):
                    ucell_delta_m = deepcopy(ucell)
                    ucell_delta_m[layer, r, c] -= delta

                    E_conv_all = to_conv_mat_discrete(ucell_delta_m, fourier_order, type_complex=type_complex)
                    o_E_conv_all = to_conv_mat_discrete(1 / ucell_delta_m, fourier_order, type_complex=type_complex)
                    de_ri_delta_m, de_ti_delta_m, _, _, _ = solver.solve(wavelength, E_conv_all, o_E_conv_all)

                    ucell_delta_p = deepcopy(ucell)
                    ucell_delta_p[layer, r, c] += delta

                    E_conv_all = to_conv_mat_discrete(ucell_delta_p, fourier_order, type_complex=type_complex)
                    o_E_conv_all = to_conv_mat_discrete(1 / ucell_delta_p, fourier_order, type_complex=type_complex)
                    de_ri_delta_p, de_ti_delta_p, _, _, _ = solver.solve(wavelength, E_conv_all, o_E_conv_all)

                    center = de_ti_delta_m.shape[0] // 2
                    grad_numeric = (de_ti_delta_p[center, center+1] - de_ti_delta_m[center, center+1]) / (2*delta)
                    grad_arr[layer, r, c] = grad_numeric

        return grad_arr

    print('Torch grad_numeric:\n', grad_numerical(ucell, 1E-6))


if __name__ == '__main__':

    optimize_jax_ucell_metasurface(1, 0, 0)
    optimize_torch_metasurface(2, 0, 0)

