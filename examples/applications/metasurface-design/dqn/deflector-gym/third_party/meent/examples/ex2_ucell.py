import time

import matplotlib.pyplot as plt
import numpy as np

from meent.rcwa import call_solver, sweep_wavelength


grating_type = 2  # 0: 1D, 1: 1D conical, 2:2D.
pol = 1  # 0: TE, 1: TM

n_I = 1  # n_incidence
n_II = 1  # n_transmission

theta = 0  # in degree, notation from Moharam paper
phi = 0  # in degree, notation from Moharam paper
psi = 0 if pol else 90  # in degree, notation from Moharam paper

wls = np.linspace(900, 900, 1)  # wavelength

if grating_type in (0, 1):
    period = [1400]
    fourier_order = 20

else:
    period = [700, 700]
    fourier_order = 3

thickness = [460, 660]

ucell = np.array([

    [
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
    ],
    [
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
    ],
])

ucell_materials = ['Si__real', 1]

AA = call_solver(mode=0, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
                 fourier_order=fourier_order, wls=wls, period=period, ucell=ucell, ucell_materials=ucell_materials,
                 thickness=thickness)
de_ri, de_ti = AA.run_ucell()
print(de_ri, de_ti)

wls = np.linspace(500, 1000, 100)

a, b = sweep_wavelength(wls, mode=0, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
                 fourier_order=fourier_order, period=period, ucell=ucell, ucell_materials=ucell_materials, thickness=thickness)

plt.plot(wls, a.sum((1, 2)), wls, b.sum((1, 2)))
plt.show()
