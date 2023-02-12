import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

os.environ["MKL_NUM_THREADS"] = "4"  # export MKL_NUM_THREADS=6

# os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
# os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

import sys
sys.path.append('/home/yongha/meent')


import jax
import jax.numpy as jnp
import numpy as np

from meent.rcwa import call_solver
from meent.on_jax.transfer_method import *


# common
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

mode_options = {0: 'numpy', 1: 'JAX', 2: 'Torch', 3: 'numpy_integ', 4: 'JAX_integ',}

a = 2
b = 1
c = 0
d = 1

device = 0
dtype = 1

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


def load_ucell(grating_type):
    if grating_type in [0, 1]:
        ucell = np.array([[[0, 0, 0, 1, 1, 1, 1, 1, 1, 1,],],])
    else:
        ucell = np.array([[
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],],])
    return ucell


ucell = load_ucell(2)
mat_list = [1, 3.48]
wavelength = 900
fourier_order = 15  # 15 20 25 30

grating_type = 2
n_I = 1
n_II = 1
period = [100, 100]
theta = 0
phi = 0
thickness = [500]
psi = 0
perturbation = 1E-10
AA = call_solver(mode=1, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi,
                     psi=psi,
                     fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell,
                     ucell_materials=ucell_materials,
                     thickness=thickness, device=0, type_complex=type_complex, )

n_iter = 3

# for i in range(n_iter):
#     t0 = time.time()
#     de_ri, de_ti = AA.run_ucell()
#     print(f'run_cell: {i}: ', time.time() - t0)

wls = [900, 901, 902]

# for i in range(n_iter):
#     t0 = time.time()
#     AA.wavelength = wls[i]
#     de_ri, de_ti = AA.run_ucell_vmap()
#     print(f'run_cell: {i}: ', time.time() - t0)

print(jax.devices())

t0 = time.time()
de_ri, de_ti = AA.run_ucell_pmap()
print(f'run_cell: ', time.time() - t0)
