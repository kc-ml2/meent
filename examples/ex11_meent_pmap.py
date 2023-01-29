import time
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import torch

from meent.rcwa import call_solver
from ex_ucell import load_ucell

from meent.on_jax.convolution_matrix import to_conv_mat, put_permittivity_in_ucell, read_material_table
from meent.on_jax.transfer_method import *
# from ex_ucell import load_ucell


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
n_iter = 2

a = 2
b = 1
c = 0
d = 1

device = 1
if device == 0:
    device = 'cpu'
    jax.config.update('jax_platform_name', device)
else:
    device = 'gpu'
    jax.config.update('jax_platform_name', device)


def load_ucell(grating_type):
    if grating_type in [0, 1]:
        ucell = np.array([[[0, 0, 0, 1, 1, 1, 1, 1, 1, 1,],],])
    else:
        ucell = jnp.array([[
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
fourier_order = 15
type_complex = jnp.complex64

grating_type = 2
n_I = 1
n_II = 1
period = [10, 10]
theta = 0
phi = 0
thickness = [10]
psi = 0
perturbation = 1E-10
AA = call_solver(mode=1, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi,
                     psi=psi,
                     fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell,
                     ucell_materials=ucell_materials,
                     thickness=thickness, device=0, type_complex=type_complex, )

for i in range(n_iter+2):
    t0 = time.time()
    de_ri, de_ti = AA.run_ucell()
    print(f'run_cell: {i}: ', time.time() - t0)
