import time
import numpy as np

from meent.on_numpy.rcwa import RCWALight as RCWA


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
    # [
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    # ],
    [
        [1, 1, 1, 3.48**2, 3.48**2, 3.48**2, 1, 1, 1, 1],
        [1, 1, 1, 3.48**2, 3.48**2, 3.48**2, 1, 1, 1, 1],
        [1, 1, 1, 3.48**2, 3.48**2, 3.48**2, 1, 1, 1, 1],
    ],
    [
        [1, 1, 1, 3.48**2, 3.48**2, 3.48**2, 1, 1, 1, 1],
        [1, 1, 1, 3.48**2, 3.48**2, 3.48**2, 1, 1, 1, 1],
        [1, 1, 1, 3.48 ** 2, 3.48 ** 2, 3.48 ** 2, 1, 1, 1, 1],
    ],
    # [
    #     [12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
    #     [12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
    #     [12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
    # ],
])

AA = RCWA(mode=0, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
                 fourier_order=fourier_order, wls=wls, period=period, ucell=ucell, thickness=thickness)
de_ri, de_ti = AA.run_ucell()
print(de_ri, de_ti)

wls = np.linspace(500, 2300, 100)
AA = RCWA(mode=0, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
                 fourier_order=fourier_order, wls=wls, period=period, ucell=ucell, thickness=thickness)
de_ri, de_ti = AA.loop_wavelength_ucell()
AA.plot()

