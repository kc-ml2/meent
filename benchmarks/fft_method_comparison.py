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

from meent.rcwa import call_solver
import torch


def load_setting(mode_key, dtype, device):
    grating_type = 2

    pol = 1  # 0: TE, 1: TM

    n_I = 1  # n_incidence
    n_II = 1  # n_transmission

    theta = 0
    phi = 0
    psi = 0 if pol else 90

    wavelength = 900

    ucell_materials = [1, 3.48]
    # thickness = [1120]
    # period = [1000, 1000]
    fourier_order = 9

    thickness, period = [1120], [100, 100]
    thickness, period = [500], [100, 100]
    thickness, period = [500], [1000, 1000]
    thickness, period = np.array([1120.]), [1000, 1000]

    ucell = np.array(
        [[
            [3., 1., 1., 1., 3.] * 2,
            [3., 1., 1., 1., 3.] * 2,
            [3., 1., 1., 1., 3.] * 2,
            [3., 1., 1., 1., 3.] * 2,
            [3., 1., 1., 1., 3.] * 2,
            [3., 1., 1., 1., 3.] * 2,
            [3., 1., 1., 1., 3.] * 2,
            [3., 1., 1., 1., 3.] * 2,
            [3., 1., 1., 1., 3.] * 2,
            [3., 1., 1., 1., 3.] * 2,
        ]]
    )
    ucell = np.array(
        [[
            [3., 1., 1., 1., 3.],
            [3., 1., 1., 1., 3.],
            [3., 1., 1., 1., 3.],
            [3., 1., 1., 1., 3.],
            [3., 1., 1., 1., 3.],
        ]]
    )
    # ucell = np.array([
    #
    #     [
    #         [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, ],
    #         [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, ],
    #         [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, ],
    #         [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, ],
    #         [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, ],
    #         [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, ],
    #         [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, ],
    #         [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, ],
    #         [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, ],
    #         [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, ],
    #     ],
    # ])

    if mode_key == 0:
        device = 0
        type_complex = np.complex128 if dtype == 0 else np.complex64
        ucell = ucell.astype(type_complex)

    elif mode_key == 1:  # JAX
        ucell = jnp.array(ucell)
        jax.config.update('jax_platform_name', 'cpu') if device == 0 else jax.config.update('jax_platform_name', 'gpu')

        if dtype == 0:
            from jax.config import config
            config.update("jax_enable_x64", True)
            type_complex = jnp.complex128
            ucell = ucell.astype(type_complex)
            # ucell = ucell.astype(jnp.float64)
        else:
            type_complex = jnp.complex64
            ucell = ucell.astype(type_complex)

    else:  # Torch
        device = torch.device('cpu') if device == 0 else torch.device('cuda')
        type_complex = torch.complex128 if dtype == 0 else torch.complex64
        ucell = torch.tensor(ucell, dtype=type_complex, device=device)
        # ucell = torch.tensor(ucell, dtype=torch.float64)

    return grating_type, pol, n_I, n_II, theta, phi, psi, wavelength, thickness, ucell_materials, period, fourier_order,\
           type_complex, device, ucell


def compare_conv_mat_method(mode_key, dtype, device):
    grating_type, pol, n_I, n_II, theta, phi, psi, wavelength, thickness, ucell_materials, period, fourier_order, \
    type_complex, device, ucell \
        = load_setting(mode_key, dtype, device)

    if mode_key == 0:
        from meent.on_numpy.convolution_matrix import to_conv_mat as conv1
        from meent.on_numpy.convolution_matrix import to_conv_mat_piecewise_constant as conv2

    elif mode_key == 1:
        from meent.on_jax.convolution_matrix import to_conv_mat as conv1
        from meent.on_jax.convolution_matrix import to_conv_mat_piecewise_constant as conv2
    else:
        from meent.on_torch.convolution_matrix import to_conv_mat as conv1
        from meent.on_torch.convolution_matrix import to_conv_mat_piecewise_constant as conv2

    for thickness, period in zip([[1120], [500], [500], [1120]], [[100, 100], [100, 100], [1000, 1000], [1000, 1000]]):

        solver = call_solver(mode_key, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi,
                             psi=psi, fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell,
                             ucell_materials=ucell_materials, thickness=thickness, device=device,
                             type_complex=type_complex, )
        E_conv_all = conv1(ucell, fourier_order, type_complex=type_complex, device=device)
        o_E_conv_all = conv1(1 / ucell, fourier_order, type_complex=type_complex, device=device)
        de_ri, de_ti, _, _, _ = solver.solve(wavelength, E_conv_all, o_E_conv_all)

        E_conv_all1 = conv2(ucell, fourier_order, type_complex=type_complex, device=device)
        o_E_conv_all1 = conv2(1 / ucell, fourier_order, type_complex=type_complex, device=device)
        de_ri1, de_ti1, _, _, _ = solver.solve(wavelength, E_conv_all1, o_E_conv_all1)

        try:
            print('de_ri, de_ti norm: ', np.linalg.norm(de_ri - de_ri1), np.linalg.norm(de_ti - de_ti1))
        except:
            print('de_ri, de_ti norm: ', torch.linalg.norm(de_ri - de_ri1),  torch.linalg.norm(de_ti - de_ti1))

    return


if __name__ == '__main__':
    t0 = time.time()

    dtype = 0
    device = 0

    compare_conv_mat_method(mode_key=0, dtype=dtype, device=device)
    compare_conv_mat_method(1, dtype=dtype, device=device)
    compare_conv_mat_method(2, dtype=dtype, device=device)
