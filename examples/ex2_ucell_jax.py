import time

import matplotlib.pyplot as plt
import jax.numpy as np
# import numpy as np
from meent.rcwa import call_solver, sweep_wavelength

grating_type = 0  # 0: 1D, 1: 1D conical, 2:2D.
pol = 1  # 0: TE, 1: TM

n_I = 1  # n_incidence
n_II = 1  # n_transmission

theta = 0.1  # in degree, notation from Moharam paper
phi = 0  # in degree, notation from Moharam paper
psi = 0 if pol else 90  # in degree, notation from Moharam paper

wavelength = np.array([900])  # wavelength

if grating_type in (0, 1):
    period = [1000]
    fourier_order = 5

    ucell = np.array([

        [
            [
                0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
            ],
        ],
        [
            [
                0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
            ],
        ],
    ])

else:
    period = [1000, 1000]
    fourier_order = 2

    ucell = np.array([

        [
            [1, 0, 0, 1, 1, 1, 1, 1, 1, 1,],
            [0, 0, 0, 1, 1, 0, 1, 1, 1, 1,],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1,],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1,],
            [1, 1, 0, 1, 1, 1, 1, 1, 1, 1,],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1,],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1,],
            [0, 1, 0, 1, 1, 1, 1, 1, 1, 1,],
            [1, 0, 0, 1, 1, 1, 1, 1, 1, 1,],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1,],
        ],
        # [
        #     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
        #     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
        #     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
        #     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
        #     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
        #     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
        #     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
        #     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
        #     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
        #     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
        # ],
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
# ucell_materials = [3.48, 1]
t0 = time.time()
AA = call_solver(mode=1, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
                 fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell, ucell_materials=ucell_materials,
                 thickness=thickness)
de_ri, de_ti = AA.run_ucell()
print(de_ri, de_ti)
print(time.time()-t0)
res = AA.calculate_field()
