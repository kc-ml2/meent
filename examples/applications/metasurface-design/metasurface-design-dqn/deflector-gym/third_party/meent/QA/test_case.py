import numpy as np

from meent.rcwa import call_solver


def test():
    pol = 1  # 0: TE, 1: TM

    n_I = 1  # n_incidence
    n_II = 1  # n_transmission

    theta = 1E-10  # in degree, notation from Moharam paper
    phi = 40  # in degree, notation from Moharam paper
    psi = 0 if pol else 90  # in degree, notation from Moharam paper

    wls = np.linspace(900, 900, 1)  # wavelength

    fourier_order = 3

    # 1D case
    period = [700]
    grating_type = 0  # 0: 1D, 1: 1D conical, 2:2D.
    thickness = [460, 660]

    ucell = np.array([
        [
            [1, 1, 1, 3.48**2, 3.48**2, 3.48**2, 1, 1, 1, 1],
        ],
        [
            [1, 1, 1, 3.48**2, 3.48**2, 3.48**2, 1, 1, 1, 1],
        ],
    ])

    AA = call_solver(mode=0, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
                     fourier_order=fourier_order, wls=wls, period=period, ucell=ucell, thickness=thickness)
    de_ri, de_ti = AA.run_ucell()
    print(de_ri, de_ti)

    # wls = np.linspace(500, 1000, 100)
    # AA = call_solver(mode=0, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
    #                  fourier_order=fourier_order, wls=wls, period=period, ucell=ucell, thickness=thickness)
    # de_ri, de_ti = AA.loop_wavelength_ucell()
    # AA.plot()


    ucell = np.array([
        [
            [n_I, n_I, n_I, n_I, n_I, n_I, n_I, n_I, n_I, n_I],
        ],
        [
            [1, 1, 1, 3.48**2, 3.48**2, 3.48**2, 1, 1, 1, 1],
        ],
        [
            [1, 1, 1, 3.48**2, 3.48**2, 3.48**2, 1, 1, 1, 1],
        ],
        [
            [n_II, n_II, n_II, n_II, n_II, n_II, n_II, n_II, n_II, n_II],
        ],
    ])

    thickness = [200, 460, 660, 200]

    wls = np.linspace(900, 900, 1)  # wavelength

    AA = call_solver(mode=0, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
                     fourier_order=fourier_order, wls=wls, period=period, ucell=ucell, thickness=thickness)
    de_ri, de_ti = AA.run_ucell()
    print(de_ri, de_ti)

    # wls = np.linspace(500, 1000, 100)
    # AA = call_solver(mode=0, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
    #                  fourier_order=fourier_order, wls=wls, period=period, ucell=ucell, thickness=thickness)
    # de_ri, de_ti = AA.loop_wavelength_ucell()
    # AA.plot()

    # 1D conical case
    period = [700]
    grating_type = 1  # 0: 1D, 1: 1D conical, 2:2D.
    thickness = [460, 660]

    ucell = np.array([
        [
            [1, 1, 1, 3.48**2, 3.48**2, 3.48**2, 1, 1, 1, 1],
        ],
        [
            [1, 1, 1, 3.48**2, 3.48**2, 3.48**2, 1, 1, 1, 1],
        ],
    ])

    wls = np.linspace(900, 900, 1)  # wavelength
    AA = call_solver(mode=0, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
                     fourier_order=fourier_order, wls=wls, period=period, ucell=ucell, thickness=thickness)
    de_ri, de_ti = AA.run_ucell()
    print(de_ri, de_ti)

    # wls = np.linspace(500, 1000, 100)
    # AA = call_solver(mode=0, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
    #                  fourier_order=fourier_order, wls=wls, period=period, ucell=ucell, thickness=thickness)
    # de_ri, de_ti = AA.loop_wavelength_ucell()
    # AA.plot()


    ucell = np.array([
        [
            [n_I, n_I, n_I, n_I, n_I, n_I, n_I, n_I, n_I, n_I],
        ],
        [
            [1, 1, 1, 3.48**2, 3.48**2, 3.48**2, 1, 1, 1, 1],
        ],
        [
            [1, 1, 1, 3.48**2, 3.48**2, 3.48**2, 1, 1, 1, 1],
        ],
        [
            [n_II, n_II, n_II, n_II, n_II, n_II, n_II, n_II, n_II, n_II],
        ],
    ])

    thickness = [200, 460, 660, 200]

    wls = np.linspace(900, 900, 1)  # wavelength

    AA = call_solver(mode=0, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
                     fourier_order=fourier_order, wls=wls, period=period, ucell=ucell, thickness=thickness)
    de_ri, de_ti = AA.run_ucell()
    print(de_ri, de_ti)

    # wls = np.linspace(500, 1000, 100)
    # AA = call_solver(mode=0, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
    #                  fourier_order=fourier_order, wls=wls, period=period, ucell=ucell, thickness=thickness)
    # de_ri, de_ti = AA.loop_wavelength_ucell()
    # AA.plot()


    # 2D case
    period = [700, 700]
    grating_type = 2  # 0: 1D, 1: 1D conical, 2:2D.
    thickness = [460, 660]

    ucell = np.array([
        [
            [1, 1, 1, 3.48**2, 3.48**2, 3.48**2, 1, 1, 1, 1],
            [1, 1, 1, 3.48**2, 3.48**2, 3.48**2, 1, 1, 1, 1],
            [1, 1, 1, 3.48**2, 3.48**2, 3.48**2, 1, 1, 1, 1],
        ],
        [
            [1, 1, 1, 3.48**2, 3.48**2, 3.48**2, 1, 1, 1, 1],
            [1, 1, 1, 3.48**2, 3.48**2, 3.48**2, 1, 1, 1, 1],
            [1, 1, 1, 3.48**2, 3.48**2, 3.48**2, 1, 1, 1, 1],
        ],
    ])

    wls = np.linspace(900, 900, 1)  # wavelength
    AA = call_solver(mode=0, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
                     fourier_order=fourier_order, wls=wls, period=period, ucell=ucell, thickness=thickness)
    de_ri, de_ti = AA.run_ucell()
    print(de_ri, de_ti)

    # wls = np.linspace(500, 1000, 100)
    # AA = call_solver(mode=0, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
    #                  fourier_order=fourier_order, wls=wls, period=period, ucell=ucell, thickness=thickness)
    # de_ri, de_ti = AA.loop_wavelength_ucell()
    # AA.plot()


    ucell = np.array([
        [
            [n_I, n_I, n_I, n_I, n_I, n_I, n_I, n_I, n_I, n_I],
            [n_I, n_I, n_I, n_I, n_I, n_I, n_I, n_I, n_I, n_I],
            [n_I, n_I, n_I, n_I, n_I, n_I, n_I, n_I, n_I, n_I],
        ],
        [
            [1, 1, 1, 3.48**2, 3.48**2, 3.48**2, 1, 1, 1, 1],
            [1, 1, 1, 3.48**2, 3.48**2, 3.48**2, 1, 1, 1, 1],
            [1, 1, 1, 3.48**2, 3.48**2, 3.48**2, 1, 1, 1, 1],
        ],
        [
            [1, 1, 1, 3.48**2, 3.48**2, 3.48**2, 1, 1, 1, 1],
            [1, 1, 1, 3.48**2, 3.48**2, 3.48**2, 1, 1, 1, 1],
            [1, 1, 1, 3.48**2, 3.48**2, 3.48**2, 1, 1, 1, 1],
        ],
        [
            [n_II, n_II, n_II, n_II, n_II, n_II, n_II, n_II, n_II, n_II],
            [n_II, n_II, n_II, n_II, n_II, n_II, n_II, n_II, n_II, n_II],
            [n_II, n_II, n_II, n_II, n_II, n_II, n_II, n_II, n_II, n_II],
        ],
    ])

    thickness = [200, 460, 660, 200]


    wls = np.linspace(900, 900, 1)  # wavelength

    AA = call_solver(mode=0, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
                     fourier_order=fourier_order, wls=wls, period=period, ucell=ucell, thickness=thickness)
    de_ri, de_ti = AA.run_ucell()
    print(de_ri, de_ti)

    # wls = np.linspace(500, 1000, 100)
    # AA = call_solver(mode=0, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
    #                  fourier_order=fourier_order, wls=wls, period=period, ucell=ucell, thickness=thickness)
    # de_ri, de_ti = AA.loop_wavelength_ucell()
    # AA.plot()
    # assert True