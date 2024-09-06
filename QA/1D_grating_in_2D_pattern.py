# This demo shows a case with 1D grating and TM polarization.
# If phi is set to 'None', this will use 1D TETM formulation (without azimuthal rotation, phi == 0)
# But if phi is set to '0', then the simulation will be taken for 1D conical or 2D case which is general but slower.

import numpy as np
from time import time

from meent import call_mee


def test():
    backend = 0
    pol = 1  # 0: TE, 1: TM

    n_top = 1  # n_incidence
    n_bot = 1  # n_transmission

    theta = 1E-10  # angle of incidence in radian

    wavelength = 300  # wavelength
    thickness = [460, 22]
    period = [700, 700]
    fto = [100, 0]

    # 1D
    ucell = np.array([
        [
            [1, 1, 1, 3.48, 3.48, 3.48, 1, 1, 1, 1],
        ],
        [
            [1, 1, 1, 3.48, 3.48, 3.48, 1, 1, 1, 1],
        ],
    ])

    mee = call_mee(backend=backend, pol=pol, n_top=n_top, n_bot=n_bot, theta=theta, fto=fto,
                   wavelength=wavelength, period=period, ucell=ucell, thickness=thickness)

    mee.phi = None
    t0_1d = time()
    res = mee.conv_solve().res
    t1_1d = time()
    de_ri1, de_ti1 = res.de_ri, res.de_ti
    print('1D (de_ri, de_ti): ', de_ri1, de_ti1)

    # 2D case
    mee.phi = 0
    t0_2d = time()
    res = mee.conv_solve().res
    t1_2d = time()
    de_ri2, de_ti2 = res.de_ri, res.de_ti
    print('2D (de_ri, de_ti): ', de_ri2, de_ti2)

    print('time for 1D formulation: ', t1_1d-t0_1d, 's')
    print('time for 2D formulation: ', t1_2d-t0_2d, 's')
    print('Simulation Difference between 1D and 2D formulation: ',
          np.linalg.norm(de_ri1 - de_ri2), np.linalg.norm(de_ti1 - de_ti2))


if __name__ == '__main__':
    test()
