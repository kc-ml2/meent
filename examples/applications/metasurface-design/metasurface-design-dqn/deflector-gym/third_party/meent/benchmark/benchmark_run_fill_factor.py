import numpy as np
# import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
import time

from interface.GRCWA import GRCWA
from interface.Reticolo import Reticolo
from meent.rcwa import call_solver

Nx = 1000
# Ny = 1001

n_I = 1
n_si = 3.48 - 7j
n_II = 1
theta = 1E-10
phi = 1E-10

fourier_order = 40

period = [700]
wls = np.linspace(500, 1000, 100)
pol = 1

thickness = [1120]
# eps for patterned layer
# pattern = np.ones(Nx, dtype=float)
pattern = np.ones(Nx, dtype=complex)
grid = np.linspace(0, period[0], Nx)

pattern[:500] = n_si
idx = np.arange(0, 500)

# pattern = np.array([n_si, 1, n_si, 1, n_si, 1, n_si, 1, n_si, 1])

textures = [n_I, [grid, pattern], n_II]

profile = np.array([[0, *thickness, 0], [1, 2, 3]])
grating_type = 0
patterns = [[np.conj(n_si), 1, 0.5]]  # n_ridge, n_groove, fill_factor

# --- Run ---

# reti
reti = Reticolo(grating_type=0, n_I=n_I, n_II=n_II, theta=theta, phi=phi, fourier_order=fourier_order,
                period=period, wls=wls, pol=pol, textures=textures, profile=profile, engine_type='octave')
t0 = time.time()
reti.run()
print('reti: ', time.time() - t0)
reti.plot(title='reticolo')

# meent TMM
meent_t = call_solver(mode=0, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi,
               fourier_order=fourier_order, wls=wls, period=[period], patterns=patterns, thickness=thickness)
t0 = time.time()

meent_t.loop_wavelength_fill_factor()
print('meent: ', time.time() - t0)

meent_t.plot(title='meent-TMM')

a = reti.spectrum_r[:, 40:40+1]
b = meent_t.spectrum_r[:, 40:40+1]

# plt.hist(a-b)
# plt.show()

print(1)
# # meent SMM
# meent_s = RCWA(grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi,
#                fourier_order=fourier_order, wls=wls, period=[period], patterns=patterns, thickness=thickness,
#                algo='SMM')
# try:
#     meent_s.loop_wavelength_fill_factor()
# except:
#     pass
# meent_s.plot(title='meent-SMM')
#
# # grcwa
# grcwa = GRCWA(grating_type=grating_type, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=0, fourier_order=fourier_order,
#               period=period, wls=wls, pol=pol, patterns=pattern ** 2, thickness=thickness)
# try:
#     grcwa.run()
# except:
#     pass
# grcwa.plot(title='grcwa')