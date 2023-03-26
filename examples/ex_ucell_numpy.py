import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

# os.environ["MKL_NUM_THREADS"] = "8"  # export MKL_NUM_THREADS=6
# os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
# os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6

import jax
import jax.numpy as jnp
import numpy as np
import time
import torch
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
period = [1000, 1000]

fourier_order = [3, 2]
mode_options = {0: 'numpy', 1: 'JAX', 2: 'Torch', }

ucell = np.array([
    [
        [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, ],
    ],
])





def run_test(grating_type, ucell, backend, dtype, device='cpu'):

    device = None

    if dtype == 0:
        type_complex = np.complex128
    else:
        type_complex = np.complex64

    mee = meent.call_mee(backend=backend, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi,
                         psi=psi, fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell,
                         thickness=thickness, device=device, type_complex=type_complex, fft_type=0, improve_dft=True)

    t0 = time.time()
    de_ri, de_ti = mee.conv_solve()
    print(f'run_cell: ', time.time() - t0)
    resolution = (20, 1, 20)

    t0 = time.time()
    field_cell = mee.calculate_field(resolution=resolution, plot=False)
    print(f'cal_field: ', time.time() - t0)

    return de_ri, de_ti, field_cell


if __name__ == '__main__':
    ucell_1d = np.array([
        [
            [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, ],
        ],
    ]) * 4 + 1

    ucell_2d = np.array([
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
    ]) * 4 + 1

    ucell = ucell * 4 + 1