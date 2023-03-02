import inspect
import jax
import torch

import jax.numpy as jnp
import numpy as np


def load_setting(mode, dtype, device, grating_type):

    pol = 1  # 0: TE, 1: TM

    n_I = 1  # n_incidence
    n_II = 1  # n_transmission

    theta = 40 * np.pi / 180
    phi = 0 * np.pi / 180
    psi = 0 * np.pi / 180 if pol else 90 * np.pi / 180

    wavelength = 900

    ucell_materials = [1, 3.48]
    fourier_order = 5
    # thickness = [1120., 400, 300]
    thickness = [1000., ]

    if grating_type in (0, 1):
        period = [1000]
        ucell = np.array(
            [
                [
                    [3.1, 1.1, 1.2, 1.6, 3.1],
                    # [3, 3, 1, 1, 1],
                ],
                # [
                #     [3.5, 1.2, 1.5, 1.2, 3.3],
                # ],
                # [
                #     [3.1, 1.5, 1.5, 1.4, 3.1],
                # ],
            ]
        )

    else:
        period = [1000, 1000]
        ucell = np.array(
            [
                # [
                #     [1, 2, 3, 4, 5],
                #     [6, 7, 8, 9, 10],
                # ],
                # [
                #     [3.5, 1.2, 1.5, 1.2, 3.3],
                #     [3.1, 1.5, 1.5, 1.4, 3.1],
                # ],
                [
                    [3.1, 1.5, 1.5, 1.4, 3.2],
                    [3.1, 1.5, 1.5, 1.8, 3.1],
                ],
            ]
        )
        ucell = np.repeat(ucell, 10, axis=1)
        ucell = np.repeat(ucell, 10, axis=2)

        # ucell = np.array(
        #     [
        #         [
        #             [3, 3, 1, 1, 1],
        #             [3, 3, 1, 1, 1],
        #             [3, 3, 1, 1, 1],
        #             [3, 3, 1, 1, 1],
        #             [3, 3, 1, 1, 1],
        #         ],
        #     ]
        # )
        # ucell = np.array(
        #     [
        #         [
        #             [3, 3, 3, 3, 3],
        #             [3, 3, 3, 3, 3],
        #             [1, 1, 1, 1, 1],
        #             [1, 1, 1, 1, 1],
        #             [1, 1, 1, 1, 1],
        #         ],
        #     ]
        # )
        # ucell = np.array(
        #     [
        #         [
        #             [1, 1, 4, 1, 1],
        #             [1, 1, 4, 1, 1],
        #             [4, 4, 4, 4, 4],
        #             [1, 1, 4, 1, 1],
        #             [1, 1, 4, 1, 1],
        #         ],
        #     ]
        # )

    if mode == 0:
        device = 0
        type_complex = np.complex128 if dtype == 0 else np.complex64
        ucell = ucell.astype(type_complex)
        thickness = np.array(thickness)

    elif mode == 1:  # JAX
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


    def retrieve_name():
        callers_local_vars = inspect.currentframe().f_back.f_locals.items()
        res = {}

        for var_name, var_val in callers_local_vars:
            if var_name in ['grating_type', 'pol', 'n_I', 'n_II', 'theta', 'phi', 'psi', 'wavelength', 'thickness', 'ucell_materials', 'period', 'fourier_order', 'type_complex', 'device', 'ucell']:
                res[var_name] = var_val

        return res

    return retrieve_name()

    # return grating_type, pol, n_I, n_II, theta, phi, psi, wavelength, thickness, ucell_materials, period, fourier_order,\
    #        type_complex, device, ucell


if __name__ == '__main__':
    aa = load_setting(0,0,0)
    print(aa)