import time
import numpy as np

from solver.BaseClass import RcwaBackbone


def run(n_I, n_II, theta, phi, grating_type, pol):

    if grating_type == 0:
        wls = np.linspace(500, 2300, 100)
        fourier_order = 40
        period = [700]
        phi = 0

    elif grating_type == 2:
        wls = np.linspace(500, 2300, 100)
        fourier_order = 2
        period = [700, 700]

    if pol == 0:
        psi = 90
    elif pol == 1:
        psi = 0

    # refractive index in grating layer
    patterns = [[3.48, 1, 0.3], [3.48, 1, 0.3]]  # n_ridge, n_groove, fill_factor
    thickness = [460, 660]

    t0 = time.time()
    res = RcwaBackbone(grating_type, n_I, n_II, theta, phi, psi, fourier_order, period, wls,
                       pol, patterns, thickness, algo='TMM')
    res.run()
    print(time.time() - t0)
    res.plot()

    t0 = time.time()
    res = RcwaBackbone(grating_type, n_I, n_II, theta, phi, psi, fourier_order, period, wls,
                       pol, patterns, thickness, algo='SMM')

    res.run()
    print(time.time() - t0)
    res.plot()

# run(2, 2, 31, 10, 2, 0)  # PASSED
# run(2, 2, theta=30, phi=10, grating_type=2, pol=0)  # SMM fail
# run(2, 2, theta=30, phi=0, grating_type=2, pol=0)  # SMM Singular but PASSED
# run(2, 2, theta=30, phi=0, grating_type=2, pol=1)  # SMM Singular but PASSED
# run(2, 2, theta=30, phi=10, grating_type=0, pol=1)  # PASS
# run(2, 2, theta=30, phi=0, grating_type=0, pol=1)  # SMM singular but PASS
# run(2, 2, theta=32, phi=10, grating_type=2, pol=1)  # PASS


# run(2, 2, theta=30, phi=0, grating_type=2, pol=1)  # SMM singular but PASS
# run(2, 2, theta=30, phi=10, grating_type=2, pol=1)  # SMM FAIL
# run(2, 2, theta=20, phi=20, grating_type=2, pol=1)  # PASS
# run(2, 2, theta=20, phi=10, grating_type=2, pol=1)  # PASS
# run(3, 2, theta=np.arcsin(1/3), phi=10, grating_type=2, pol=1)  # PASS

# run(2, 2, theta=30, phi=10, grating_type=2, pol=1)  # SMM FAIL
# run(2, 2, theta=31, phi=10, grating_type=2, pol=1)  # PASS

# run(2, 2, theta=30, phi=10, grating_type=0, pol=1)  # SMM FAIL
# run(2, 2, theta=31, phi=10, grating_type=0, pol=1)  # singular but pass

run(2, 2, theta=30+1E-10, phi=10, grating_type=2, pol=0)  # SMM weak FAIL
# run(2, 2, theta=31+1E-10, phi=10, grating_type=2, pol=0)  # pass

run(2, 2, theta=30, phi=10, grating_type=2, pol=0)  # SMM FAIL
# run(2, 2, theta=31, phi=10, grating_type=2, pol=0)  # singular but pass

run(2, 2, theta=30+1E-10, phi=10, grating_type=2, pol=1)  # SMM weak FAIL
# run(2, 2, theta=31+1E-10, phi=10, grating_type=2, pol=1)  # PASS

run(2, 2, theta=30, phi=10, grating_type=2, pol=1)  # SMM FAIL
# run(2, 2, theta=31, phi=10, grating_type=2, pol=1)  # PASS

