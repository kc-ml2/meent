import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from interface.GRCWA import GRCWA
from interface.Reticolo import Reticolo
from meent.rcwa import RCWA

Nx = 1001
Ny = 1001

n_I = 1
n_si = 3.48
n_II = 1
theta = 1
phi = 0
psi = 90

fourier_order = 40

period = 700
wls = np.linspace(500, 2300, 100)
pol = 0

thickness = 1120
# eps for patterned layer
pattern = np.ones(Nx, dtype=float)
grid = np.linspace(0, period, 1001)
pattern[:300] = n_si

textures = [n_I, [grid, pattern], n_II]

profile = np.array([[0, thickness, 0], [1, 2, 3]])
grating_type = 0
patterns = [[3.48, 1, 0.3]]  # n_ridge, n_groove, fill_factor
thickness = [1120]
# --- Run ---

# reti
reti = Reticolo(grating_type=0, n_I=n_I, n_II=n_II, theta=theta, phi=phi, fourier_order=fourier_order,
                period=period, wls=wls, pol=pol, textures=textures, profile=profile, engine_type='octave')
reti.run()
reti.plot(title='reticolo')

# grcwa
grcwa = GRCWA(grating_type=grating_type, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=0, fourier_order=fourier_order,
              period=period, wls=wls, pol=pol, patterns=pattern ** 2, thickness=thickness)
grcwa.run()
grcwa.plot(title='grcwa')

# meent TMM
meent_t = RCWA(grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
               fourier_order=fourier_order, wls=wls, period=[period], patterns=patterns, thickness=thickness)
meent_t.loop_wavelength()
meent_t.plot(title='meent-TMM')
# meent SMM
meent_s = RCWA(grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
               fourier_order=fourier_order, wls=wls, period=[period], patterns=patterns, thickness=thickness,
               algo='SMM')
meent_s.loop_wavelength()
meent_s.plot(title='meent-SMM')
