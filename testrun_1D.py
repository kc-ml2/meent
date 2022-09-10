import numpy as np
import matplotlib.pyplot as plt

from solver.LalanneClass import LalanneBase


n_I = 1
n_II = 1

theta = 0
phi = 0
psi = 0

fourier_order = 3
period = [0.7]

wls = np.linspace(0.5, 2.3, 400)

polarization = 1  # TE 0, TM 1

# permittivity in grating layer
# patterns = [[3.48, 1, 0.3], [3.48, 1, 0.3]]  # n_ridge, n_groove, fill_factor
patterns = [['SILICON', 1, 0.3]]  # n_ridge, n_groove, fill_factor
thickness = [0.46]

polarization_type = 0

res = LalanneBase(polarization_type, n_I, n_II, theta, phi, psi, fourier_order, period, wls, polarization, patterns,
                  thickness)

res.lalanne_1d()
# res.lalanne_1d_conical()
# res.lalanne_2d()

plt.plot(res.wls, res.spectrum_r)
plt.plot(res.wls, res.spectrum_t)
plt.show()
