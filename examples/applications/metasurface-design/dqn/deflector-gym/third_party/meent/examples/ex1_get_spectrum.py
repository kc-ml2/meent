import time
import numpy as np

from meent.rcwa import call_solver

grating_type = 0  # 0: 1D, 1: 1D conical, 2:2D.
pol = 1  # 0: TE, 1: TM

n_I = 1  # n_incidence
n_II = 1  # n_transmission

theta = 0  # in degree, notation from Moharam paper
phi = 0  # in degree, notation from Moharam paper
psi = 0 if pol else 90  # in degree, notation from Moharam paper

wls = np.linspace(500, 1000, 100)  # wavelength

if grating_type in (0, 1):
    period = [700]
    fourier_order = 20
    patterns = [[3.48, 1, 0.3], [3.48, 1, 0.3]]  # n_ridge, n_groove, fill_factor

else:
    period = [700, 700]
    fourier_order = 2
    patterns = [[3.48, 1, [0.3, 1]], [3.48, 1, [0.3, 1]]]  # n_ridge, n_groove, fill_factor[x, y]

thickness = [460, 660]

t0 = time.perf_counter()
solver = call_solver(mode=0, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
                 fourier_order=fourier_order, wls=wls, period=period, patterns=patterns, thickness=thickness)

a, b = solver.loop_wavelength_fill_factor()
solver.plot()

print('wall time: ', time.perf_counter() - t0)
