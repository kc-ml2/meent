import time

import matplotlib.pyplot as plt
import numpy as np

from meent.rcwa import call_solver

grating_type = 2  # 0: 1D, 1: 1D conical, 2:2D.
pol = 1  # 0: TE, 1: TM

n_I = 1  # n_incidence
n_II = 1  # n_transmission

theta = 0 * np.pi / 180
phi = 0 * np.pi / 180
psi = 0 if pol else 90 * np.pi / 180  # in degree, notation from Moharam paper

wavelength = 900

period = [700, 700]
fourier_order = 2

thickness = [100] * 10

ucell_z, ucell_y, ucell_x = 10, 100, 100

ucell = np.zeros((ucell_z, ucell_y, ucell_x), dtype=int)

z, y, x = np.indices((ucell_z, ucell_y, ucell_x))

x = x - ucell_x // 2
y = y - ucell_y // 2

colors = np.empty([*ucell.shape, 4], dtype=object)
voxelarray = np.full(ucell.shape, False)

parts = {}
for i in range(ucell_z):
    parts['base'] = 0
    indices = (x ** 2 + y ** 2 < ((5 * i // 2) + 20) ** 2) & (z == i)
    parts[f'hole{i}'] = indices
    ucell[indices] = 1
    colors[indices] = (0., 1., 0., 1)
    voxelarray |= indices

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.voxels(voxelarray.swapaxes(0, 2), facecolors=colors.swapaxes(0, 2))
plt.show()

plt.imshow(np.flipud(ucell[:, 50, :]), aspect='auto')
plt.show()

ucell_materials = ['p_si__real', 1]

solver = call_solver(mode=0, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
                     fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell,
                     ucell_materials=ucell_materials, thickness=thickness)

wavelength_array = np.linspace(500, 1000, 100)

spectrum_r = np.zeros([len(wavelength_array)] + [2 * fourier_order + 1] * (grating_type // 2 + 1))
spectrum_t = np.zeros([len(wavelength_array)] + [2 * fourier_order + 1] * (grating_type // 2 + 1))

for i, wavelength in enumerate(wavelength_array):
    solver.wavelength = wavelength
    de_ri, de_ti = solver.run_ucell()

    spectrum_r[i] = de_ri
    spectrum_t[i] = de_ti

plt.plot(wavelength_array, spectrum_r.sum((1, 2)), wavelength_array, spectrum_t.sum((1, 2)))
plt.show()
