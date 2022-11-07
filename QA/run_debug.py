import time
import numpy as np

from meent.rcwa import RCWA


def run_debug_cases(n_I, n_II, theta, phi, grating_type, pol):

    if grating_type == 0:
        wls = np.linspace(500, 2300, 100)
        fourier_order = 40
        period = [700]
        patterns = [[3.48, 1, 0.3], [3.48, 1, 0.3]]  # n_ridge, n_groove, fill_factor
        phi = 0


    elif grating_type == 1:
        wls = np.linspace(500, 2300, 100)
        fourier_order = 10
        period = [700]
        patterns = [[3.48, 1, 0.3], [3.48, 1, 0.3]]  # n_ridge, n_groove, fill_factor

    elif grating_type == 2:
        wls = np.linspace(500, 2300, 100)
        fourier_order = 2
        period = [700, 700]
        patterns = [[3.48, 1, [0.3, 1]], [3.48, 1, [0.3, 1]]]  # n_ridge, n_groove, fill_factor

    if pol == 0:
        psi = 90
    elif pol == 1:
        psi = 0

    # refractive index in grating layer
    thickness = [460, 660]

    t0 = time.time()
    res = RCWA(grating_type, n_I, n_II, theta, phi, psi, fourier_order, period, wls,
                        pol, patterns, thickness, algo='TMM')
    res.loop_wavelength_fill_factor()
    print(time.time() - t0)
    res.plot()

    # t0 = time.time()
    # res = RCWA(grating_type, n_I, n_II, theta, phi, psi, fourier_order, period, wavelength,
    #                     pol, patterns, thickness, algo='SMM')
    #
    # res.loop_wavelength_fill_factor()
    # print(time.time() - t0)
    # res.plot()

# run_debug_cases(2, 2, 31, 10, 2, 0)  # PASSED
# run_debug_cases(2, 2, theta=30, phi=10, grating_type=2, pol=0)  # SMM fail
# run_debug_cases(2, 2, theta=30, phi=0, grating_type=2, pol=0)  # SMM Singular but PASSED
# run_debug_cases(2, 2, theta=30, phi=0, grating_type=2, pol=1)  # SMM Singular but PASSED
# run_debug_cases(2, 2, theta=30, phi=10, grating_type=0, pol=1)  # PASS
# run_debug_cases(2, 2, theta=30, phi=0, grating_type=0, pol=1)  # SMM singular but PASS
# run_debug_cases(2, 2, theta=32, phi=10, grating_type=2, pol=1)  # PASS


# run_debug_cases(2, 2, theta=30, phi=0, grating_type=2, pol=1)  # SMM singular but PASS
# run_debug_cases(2, 2, theta=30, phi=10, grating_type=2, pol=1)  # SMM FAIL
# run_debug_cases(2, 2, theta=20, phi=20, grating_type=2, pol=1)  # PASS
# run_debug_cases(2, 2, theta=20, phi=10, grating_type=2, pol=1)  # PASS
# run_debug_cases(3, 2, theta=np.arcsin(1/3), phi=10, grating_type=2, pol=1)  # PASS

# run_debug_cases(2, 2, theta=30, phi=10, grating_type=2, pol=1)  # SMM FAIL
# run_debug_cases(2, 2, theta=31, phi=10, grating_type=2, pol=1)  # PASS

# run_debug_cases(2, 2, theta=30, phi=10, grating_type=0, pol=1)  # SMM FAIL
# run_debug_cases(2, 2, theta=31, phi=10, grating_type=0, pol=1)  # singular but pass

run_debug_cases(2, 2, theta=30 + 1E-10, phi=10, grating_type=2, pol=0)  # SMM weak FAIL
# run_debug_cases(2, 2, theta=31+1E-10, phi=10, grating_type=2, pol=0)  # pass

run_debug_cases(2, 2, theta=30, phi=10, grating_type=2, pol=0)  # SMM FAIL
# run_debug_cases(2, 2, theta=31, phi=10, grating_type=2, pol=0)  # singular but pass

run_debug_cases(2, 2, theta=30 + 1E-10, phi=10, grating_type=2, pol=1)  # SMM weak FAIL
# run_debug_cases(2, 2, theta=31+1E-10, phi=10, grating_type=2, pol=1)  # PASS

run_debug_cases(2, 2, theta=30, phi=10, grating_type=2, pol=1)  # SMM FAIL
# run_debug_cases(2, 2, theta=31, phi=10, grating_type=2, pol=1)  # PASS

