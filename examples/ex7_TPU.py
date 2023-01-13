import jax
import jax.numpy as jnp
try:
    import jax.tools.colab_tpu
    jax.tools.colab_tpu.setup_tpu()
except:
    pass
# jax.device_count()

if 0:
    import drive.MyDrive.meent
    from drive.MyDrive.meent.rcwa import call_solver
else:
    import meent
    from meent.rcwa import call_solver


import time

import matplotlib.pyplot as plt
import numpy as np
import torch

# from ex2_ucell_functions import get_cond_numpy, get_cond_jax, get_cond_torch
# from ex2_ucell_functions import get_cond_numpy, get_cond_torch


def get_cond_jax(grating_type, type_int=jnp.int32):
    fourier_order = 1
    if grating_type in [0, 1]:
        period = [1000]
        ucell = jnp.array([
            [
                [
                    0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                ],
            ],
        ], dtype=type_int)
    else:
        period = [1000, 1000]
        ucell = jnp.array([
            [
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
            ],
        ], dtype=type_int)
    return period, fourier_order, ucell


# grating_type = 1  # 0: 1D, 1: 1D conical, 2:2D.
pol = 1  # 0: TE, 1: TM

n_I = 1  # n_incidence
n_II = 1  # n_transmission

theta = 1E-10
phi = 0
psi = 0 if pol else 90

wavelength = 900

thickness = [500]
ucell_materials = [1, 3.48]

mode_options = {0: 'numpy', 1: 'JAX', 2: 'Torch', 3: 'numpy_integ', 4: 'JAX_integ', }
n_iter = 2


def run_test(grating_type, mode_key, dtype, device):
    # Numpy
    if mode_key == 0:

        if device != 0:
            return

        if dtype == 0:
            type_complex = np.complex128
        elif dtype == 1:
            type_complex = np.complex64

        period, fourier_order, ucell = get_cond_numpy(grating_type)

    # JAX
    elif mode_key == 1:
        import jax

        if device == 0:
            jax.config.update('jax_platform_name', 'cpu')
        elif device == 1:
            jax.config.update('jax_platform_name', 'gpu')
        elif device == 2:
            import jax.tools.colab_tpu
            jax.tools.colab_tpu.setup_tpu()

        if dtype == 0:
            # from jax.config import config
            # config.update("jax_enable_x64", True)
            jax.config.update("jax_enable_x64", True)

            type_complex = jnp.complex128

        elif dtype == 1:
            # from jax.config import config
            # config.update("jax_enable_x64", False)
            jax.config.update("jax_enable_x64", False)
            type_complex = jnp.complex64
        else:
            return
        period, fourier_order, ucell = get_cond_jax(grating_type)

    else:
        # Torch
        if device == 0:
            device = torch.device('cpu')
        elif device == 1:
            device = torch.device('cuda')
        else:
            return

        if dtype == 0:
            type_complex = torch.complex128
        elif dtype == 1:
            type_complex = torch.complex64
        else:
            return
        period, fourier_order, ucell = get_cond_torch(grating_type)

    AA = call_solver(mode=mode_key, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi,
                     psi=psi,
                     fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell,
                     ucell_materials=ucell_materials,
                     thickness=thickness, device=device, type_complex=type_complex, )
    for i in range(n_iter):
        t0 = time.time()
        de_ri, de_ti = AA.run_ucell()
        print(f'run_cell: {i}: ', time.time() - t0)

    resolution = (20, 20, 20)
    for i in range(1):
        t0 = time.time()
        AA.calculate_field(resolution=resolution, plot=False)
        print(f'cal_field: {i}', time.time() - t0)

    return de_ri, de_ti


def run_loop():
    for grating_type in [0, 1, 2]:
        for bd in [1]:
            for dtype in [0, 1, 2]:
                for device in [0, 1, 2]:
                    print(f'grating:{grating_type}, backend:{bd}, dtype:{dtype}, dev:{device}')
                    run_test(grating_type, bd, dtype, device)

                    # try:
                    #     print(f'grating:{grating_type}, backend:{bd}, dtype:{dtype}, dev:{device}')
                    #     run_test(grating_type, bd, dtype, device)
                    # except Exception as e:
                    #     print(e)


def run_assert():
    for grating_type in [0, 1, 2]:
        for bd in [0, 1, 2]:
            print(run_test(grating_type, bd, 0, 0))

