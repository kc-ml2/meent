import numpy as np

from meent.main import call_mee


def test():
    backend = 0
    pol = 1  # 0: TE, 1: TM

    n_top = 1  # n_incidence
    n_bot = 1  # n_transmission

    theta = 1E-10  # angle of incidence in radian
    phi = 0  # azimuth angle in radian

    wavelength = 300  # wavelength
    thickness = [460, 22]
    period = [700, 700]
    fto = [10, 0]

    # 1D
    ucell = np.array([
        [
            [1, 1, 1, 3.48, 3.48, 3.48, 1, 1, 1, 1],
        ],
        [
            [1, 1, 1, 3.48, 3.48, 3.48, 1, 1, 1, 1],
        ],
    ])

    AA = call_mee(backend=backend, pol=pol, n_top=n_top, n_bot=n_bot, theta=theta, phi=phi,
                  fto=fto, wavelength=wavelength, period=period, ucell=ucell, thickness=thickness)
    de_ri, de_ti = AA.conv_solve()
    print('1D', de_ri.sum(), de_ti.sum())

    # 2D case

    ucell = np.array([
        [
            [1, 1, 1, 3.48, 3.48, 3.48, 1, 1, 1, 1],
            [1, 1, 1, 3.48, 3.48, 3.48, 1, 1, 1, 1],
            [1, 1, 1, 3.48, 3.48, 3.48, 1, 1, 1, 1],
        ],
        [
            [1, 1, 1, 3.48, 3.48, 3.48, 1, 1, 1, 1],
            [1, 1, 1, 3.48, 3.48, 3.48, 1, 1, 1, 1],
            [1, 1, 1, 3.48, 3.48, 3.48, 1, 1, 1, 1],
        ],
    ])

    AA = call_mee(backend=backend, pol=pol, n_top=n_top, n_bot=n_bot, theta=theta, phi=phi,
                  fto=fto, wavelength=wavelength, period=period, ucell=ucell, thickness=thickness)
    de_ri, de_ti = AA.conv_solve()
    print('2D', de_ri.sum(), de_ti.sum())


if __name__ == '__main__':
    test()
