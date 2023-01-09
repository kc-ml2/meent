import time

import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from meent.rcwa import call_solver, sweep_wavelength
import jax
import torch

# JAX
jax.config.update('jax_platform_name', 'cpu')
# jax.config.update('jax_platform_name', 'gpu')

# Torch
device = torch.device('cuda')
# device = torch.device('cpu')

type_complex = torch.complex128
type_complex = torch.complex64
# type_complex = np.complex64
# type_complex = jnp.complex64

# common
grating_type = 2  # 0: 1D, 1: 1D conical, 2:2D.
pol = 1  # 0: TE, 1: TM

n_I = 1  # n_incidence
n_II = 1  # n_transmission

theta = 0.1
phi = 0
psi = 0 if pol else 90
# wavelength = np.array([900])
wavelength = 900  # TODO: in numpy mode, np.array([900]) and 900 shows different result. final result of array shows 1E-14 order but 900 only shows 0


if grating_type in (0, 1):
    period = [1000]
    fourier_order = 20

    ucell = np.array([

        [
            [
                0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
            ],
        ],
        # [
        #     [
        #         0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
        #     ],
        # ],
    ])

else:
    # period = torch.tensor([1000, 1000])
    period = [1000, 1000]
    fourier_order = 15
    fourier_order = 20

    # ucell = torch.tensor([
    ucell = np.array([
        # [
        #     [0, 0, 0, 1, 1, 1, 1, 0, 0, 0,],
        #     [0, 0, 0, 1, 1, 0, 1, 1, 1, 1,],
        #     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1,],
        #     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1,],
        #     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1,],
        #     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1,],
        #     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1,],
        #     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1,],
        #     [1, 0, 0, 1, 1, 1, 1, 1, 1, 1,],
        #     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1,],
        # ],

        # [
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        #     [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, ],
        #     [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, ],
        #     [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, ],
        #     [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, ],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        # ],

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

    # ucell = np.array([
    #
    #     [
    #         [0, 0, 0, 1, 1, 1, 1, 1, 1, 1,],
    #         [0, 0, 0, 1, 1, 1, 1, 1, 1, 1,],
    #         [0, 0, 0, 1, 1, 1, 1, 1, 1, 1,],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,],
    #     ],
    # ])


thickness = [500]


ucell_materials = [1, 3.48]

mode_options = {0: 'numpy', 1: 'JAX', 2: 'Torch', 3: 'numpy_integ', 4: 'JAX_integ',}

mode_key = 2

n_iter = 1

print(mode_options[mode_key])

AA = call_solver(mode=mode_key, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
                 fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell,
                 ucell_materials=ucell_materials,
                 thickness=thickness, device=device, type_complex=type_complex, )

for i in range(n_iter):
    t0 = time.time()
    de_ri, de_ti = AA.run_ucell()
    print(f'run {i}: ', time.time()-t0)


resolution = (20, 20, 20)
for i in range(1):
    t0 = time.time()
    AA.calculate_field(resolution=resolution, plot=True)
    print(time.time() - t0)


# print(de_ri, de_ti)
