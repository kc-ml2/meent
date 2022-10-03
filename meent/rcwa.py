import time
import numpy as np


def call_solver(mode=0, *args, **kwargs):
    if mode == 0:
        from meent.on_numpy.rcwa import RCWALight
        RCWA = RCWALight(mode, *args, **kwargs)
    elif mode == 1:
        from meent.on_jax.rcwa import RCWAOpt
        RCWA = RCWAOpt(mode, *args, **kwargs)
    else:
        raise ValueError

    return RCWA


if __name__ == '__main__':
    grating_type = 0
    pol = 0

    n_I = 1
    n_II = 1

    theta = 0
    phi = 0
    psi = 0 if pol else 90

    wls = np.linspace(500, 1300, 100)
    # wls = np.linspace(600, 800, 3)

    if grating_type in (0, 1):
        period = [700]
        patterns = [[3.48, 1, 0.1], [3.48, 1, 0.1]]  # n_ridge, n_groove, fill_factor
        fourier_order = 40

    elif grating_type == 2:
        period = [700, 700]
        patterns = [[3.48, 1, [0.3, 1]], [3.48, 1, [0.3, 1]]]  # n_ridge, n_groove, fill_factor[x, y]
        fourier_order = 2
    else:
        raise ValueError

    thickness = [460, 660]

    mode = 0  # 0: light mode; 1: backprop mode;

    # AA = call_solver(mode=mode, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
    #           fourier_order=fourier_order, wls=wls, period=period, patterns=patterns, thickness=thickness)
    # t0 = time.perf_counter()
    #
    # a, b = AA.loop_wavelength_fill_factor()
    # AA.plot()
    # print(time.perf_counter() - t0)

    ucell = np.array(
        [
            [
                [3.48 ** 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
            [
                [3.48 ** 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ],
        ]
    )

    AA = call_solver(mode=mode, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
              fourier_order=fourier_order, wls=wls, period=period, patterns=patterns, ucell = ucell, thickness=thickness)
    t0 = time.perf_counter()

    a, b = AA.loop_wavelength_ucell()
    AA.plot()

    print(time.perf_counter() - t0)
