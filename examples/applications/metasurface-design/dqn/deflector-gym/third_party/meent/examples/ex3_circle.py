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

period = [700, 700]
fourier_order = 2

thickness = [460, 660]

ucell_x, ucell_y = 100, 100

ucell = np.zeros((2, ucell_x, ucell_y), dtype=int)

x = np.arange(ucell_x) - ucell_x // 2
y = np.arange(ucell_y) - ucell_y // 2

hole = x**2 + y[:, None]**2 < 20 ** 2
ucell[0, hole] = 1

hole = x**2 + y[:, None]**2 < 15 ** 2
ucell[1, hole] = 1

plt.imshow(ucell[0])
plt.show()
plt.imshow(ucell[1])
plt.show()

ucell_materials = [1, 'p_si', 'p_si__real']

AA = call_solver(mode=0, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
                 fourier_order=fourier_order, wls=wls, period=period, ucell=ucell, ucell_materials=ucell_materials,
                 thickness=thickness)
de_ri, de_ti = AA.run_ucell()
print(de_ri, de_ti)

wls = np.linspace(500, 1000, 10)

de_ri, de_ti = sweep_wavelength(wls, mode=0, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
                 fourier_order=fourier_order, period=period, ucell=ucell, ucell_materials=ucell_materials, thickness=thickness)

plt.plot(wls, de_ri.sum((1, 2)), wls, de_ti.sum((1, 2)))
plt.show()
