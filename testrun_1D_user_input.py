import time

import numpy as np
import matplotlib.pyplot as plt

from solver.LalanneClass import LalanneBase
from solver.JLAB import JLABCode


def run_mcwa(pattern_pixel, wavelength, trans_angle):
    n_I = 1.45
    n_II = 1  # glass

    theta = 0
    phi = 0
    psi = 0

    fourier_order = 40

    wls = np.array([wavelength])
    if len(wls) != 1:
        raise ValueError('wavelength should be a single value')

    period = abs(wls / np.sin(trans_angle / 180 * np.pi))

    grating_type = 0  # 1D 0, 1D_conical 1, 2D 2;
    # TODO: not implemented

    # permittivity in grating layer
    patterns = [['SILICON', 1, pattern_pixel]]  # n_ridge, n_groove, pattern_pixel

    thickness = [325]

    polarization = 1

    aa = JLABCode(grating_type, n_I, n_II, theta, phi, psi, fourier_order, period, wls, polarization, patterns,
                      thickness)

    tran_cut = aa.run_1d()

    return tran_cut[0][0], tran_cut[0]


if __name__ == '__main__':
    t0 = time.perf_counter()
    pattern = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
               1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
               1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
               1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
               1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., -1.])
    pattern = np.array([-1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
        -1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
         1.,  1.,  1.,  1.,  1.,  1.,  1.,  1., -1.,  1.,  1.,  1.,  1.,
         1.,  1.,  1.,  1.,  1.,  1., -1.,  1.,  1.,  1.,  1.,  1.,  1.,
         1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1., -1.,  1.,  1.])
    a, b= run_mcwa(pattern, 900, 60)
    print(time.perf_counter() - t0)

