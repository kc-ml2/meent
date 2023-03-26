# import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'

# os.environ["MKL_NUM_THREADS"] = "8"  # export MKL_NUM_THREADS=6
# os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
# os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6

import jax
import jax.numpy as jnp
import numpy as np
import time

import meent

# common
pol = 0  # 0: TE, 1: TM

n_I = 1  # n_incidence
n_II = 1  # n_transmission

theta = 20 * np.pi / 180
phi = 50 * np.pi / 180
psi = 0 if pol else 90 * np.pi / 180

wavelength = 900

thickness = [500]
ucell_materials = [1, 'p_si__real']
period = [1000, 1000]

fourier_order = [3, 3]
mode_options = {0: 'numpy', 1: 'JAX', 2: 'Torch', }
n_iter_de = 1
n_iter_field = 2


def run_test(grating_type, mode_key, dtype, device):
    ucell = load_ucell(grating_type)

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

    from meent.on_jax.modeler.modeling import ModelingJax
    ucell = ModelingJax().put_refractive_index_in_ucell(ucell, ucell_materials, wavelength, type_complex)

    mee = meent.call_mee(mode=mode_key, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi,
                        psi=psi, fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell,
                        ucell_materials=ucell_materials,
                        thickness=thickness, device=device, type_complex=type_complex, fft_type=0, improve_dft=True)

    for i in range(n_iter_de):
        t0 = time.time()
        mee.conv_solve()
        print(f'run_cell: {i}: ', time.time() - t0)
    resolution = (20, 30, 40)

    for i in range(n_iter_field):

        t0 = time.time()
        mee.calculate_field_all(resolution=resolution, plot=True)
        print(f'cal_field: {i}', time.time() - t0)


def run_loop(a, b, c, d):
    for grating_type in a:
        for bd in b:
            for dtype in c:
                for device in d:
                    print(f'grating:{grating_type}, backend:{bd}, dtype:{dtype}, dev:{device}')
                    run_test(grating_type, bd, dtype, device)
                    print('\n')


def load_ucell(grating_type):
    if grating_type in [0, 1]:

        ucell = np.array([

            [
                [
                    0, 1, 0, 1, 1, 0, 1, 0, 1, 1,
                ],
            ],
        ])
    else:

        ucell = np.array([
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
        ])

        ucell = np.array([
            [
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, ],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, ],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, ],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, ],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, ],
            ],
        ])

    return ucell


if __name__ == '__main__':
    run_loop([2], [1], [0], [0])