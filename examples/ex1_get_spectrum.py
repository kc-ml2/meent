import time
import numpy as np

from meent.rcwa import RCWA


grating_type = 0  # 0: 1D, 2:2D.
pol = 0  # 0: TE, 1: TM

n_I = 1  # n_incidence
n_II = 1  # n_transmission

theta = 1E-10
phi = 0
psi = 0 if pol else 90

wls = np.linspace(500, 2300, 100)  # wavelength

if grating_type == 2:
    period = [700, 700]
    fourier_order = 5

else:
    period = [700]
    fourier_order = 2

# permittivity in grating layer
patterns = [[3.48, 1, 0.3], [3.48, 1, 0.2]]  # n_ridge, n_groove, fill_factor
thickness = [460, 660]

AA = RCWA(grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
          fourier_order=fourier_order, wls=wls, period=period, patterns=patterns, thickness=thickness)
t0 = time.perf_counter()

a, b = AA.loop_wavelength_fill_factor()
AA.plot()

print(time.perf_counter() - t0)

