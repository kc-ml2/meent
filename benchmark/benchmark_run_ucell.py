import numpy as np
# import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
import time

from interface.GRCWA import GRCWA
from interface.Reticolo import Reticolo
from meent.rcwa import call_solver


n_I = 1
n_si = 3.48
n_air = 1
n_II = 1
theta = 1E-10
phi = 1E-10

fourier_order = 40

period = [700]
wavelength = np.array([900.])
pol = 1
psi = 0 if pol else 90  # in degree, notation from Moharam paper

thickness = [1120]

pattern = np.array([n_si, n_si, n_si, n_air, n_air, n_air, n_air, n_air, n_air, n_air, ])

N = len(pattern)
dx = period[0]/N
grid = np.arange(1, N+1)*dx

textures = [n_I, [grid, pattern], n_II]

profile = np.array([[0, *thickness, 0], [1, 2, 3]])
grating_type = 0

pattern = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, ])
ucell_materials = [n_air, n_si]

ucell = np.array([[pattern]])

# --- Run ---

# reti
reti = Reticolo(grating_type=0, n_I=n_I, n_II=n_II, theta=theta, phi=phi, fourier_order=fourier_order,
                period=period, wavelength=wavelength, pol=pol, textures=textures, profile=profile, engine_type='octave')
# t0 = time.time()
reti.run()
# print('reti: ', time.time() - t0)
# reti.plot(title='reticolo')

# meent TMM
meent_t = call_solver(mode=0, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
                 fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell, ucell_materials=ucell_materials,
                 thickness=thickness)
t0 = time.time()
de_ri, de_ti = meent_t.run_ucell()
print('difference:', (reti.spectrum_r[0] - de_ri).sum())
