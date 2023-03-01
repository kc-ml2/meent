import os
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))

import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

# os.environ["MKL_NUM_THREADS"] = "48"  # export MKL_NUM_THREADS=6

# os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
# os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6

# os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=24'


import numpy as np

from meent.entrance import call_solver
from meent.on_jax.transfer_method import *


# common
# grating_type = 1  # 0: 1D, 1: 1D conical, 2:2D.
pol = 1  # 0: TE, 1: TM

n_I = 1  # n_incidence
n_II = 1  # n_transmission

theta = 0 * np.pi / 180
phi = 0 * np.pi / 180
psi = 0 * np.pi / 180 if pol else 90 * np.pi / 180

wavelength = 900

thickness = [500]
ucell_materials = [1, 3.48]

mode_options = {0: 'numpy', 1: 'JAX', 2: 'Torch', }
mat_list = [1, 3.48]
fourier_order = 5  # 15 20 25 30

grating_type = 2

period = [1000, 1000]
perturbation = 1E-10

device = 1
dtype = 0

if device == 0:
    device = 'cpu'
    jax.config.update('jax_platform_name', device)
else:
    device = 'gpu'
    jax.config.update('jax_platform_name', device)

if dtype == 0:
    jax.config.update('jax_enable_x64', True)
    type_complex = jnp.complex128
else:
    type_complex = jnp.complex64

solver = call_solver(mode=1, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi,
                     psi=psi, fourier_order=fourier_order, wavelength=wavelength, period=period,
                     ucell_materials=ucell_materials,
                     thickness=thickness, device=device, type_complex=type_complex, )

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
], dtype=np.float64)

ucell *= 2.48
ucell += 1.

ucell1 = ucell.copy()
ucell2 = ucell.copy()
ucell3 = ucell.copy()

ucell1[0, :, 0] = 1
ucell2[0, :, 1] = 1
ucell3[0, :, 2] = 1

ucell_list = np.array([ucell1]*2)

for i, ucell in enumerate(ucell_list[:2]):
    t0 = time.time()
    de_ri, de_ti = solver.conv_solve(ucell)
    print(time.time() - t0)

# for i, ucell in enumerate(ucell_list[:2]):
#     t0 = time.time()
#     de_ri, de_ti = emsolver.conv_solve(ucell)
#     print(time.time() - t0)

# t0 = time.time()
# de_ri, de_ti = emsolver.run_ucell_vmap(ucell_list)
# print(f'vmap: ', time.time() - t0)
#
# t0 = time.time()
# de_ri, de_ti = emsolver.run_ucell_vmap(ucell_list)
# print(f'vmap: ', time.time() - t0)

t0 = time.time()
de_ri, de_ti = solver.run_ucell_pmap(ucell_list)
print(time.time() - t0)

t0 = time.time()
de_ri, de_ti = solver.run_ucell_pmap(ucell_list)
print(time.time() - t0)
