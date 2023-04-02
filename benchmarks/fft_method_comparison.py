import numpy as np
import time

from meent.main import call_mee
import torch


def load_setting():
    grating_type = 2

    pol = 0  # 0: TE, 1: TM

    n_I = 1  # n_incidence
    n_II = 1.5  # n_transmission

    theta = 0
    phi = 0

    wavelength = 900

    thickness = [1120]
    period = [1000, 1000]
    fourier_order = [5, 5]

    ucell = np.array([

        [
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, ],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, ],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, ],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, ],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, ],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, ],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, ],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, ],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, ],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, ],
        ],
    ]) * 4 + 1

    return grating_type, pol, n_I, n_II, theta, phi, wavelength, thickness, period, fourier_order,\
        ucell


def compare_conv_mat_method(backend, type_complex, device):
    grating_type, pol, n_I, n_II, theta, phi, wavelength, thickness, period, fourier_order, ucell = load_setting()

    for thickness, period in zip([[1120], [500], [500], [1120]], [[100, 100], [100, 100], [1000, 1000], [1000, 1000]]):

        mee = call_mee(backend, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi,
                       fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell,
                       ucell_thickness=thickness, device=device,
                       type_complex=type_complex, )

        mee.fft_type = 0
        de_ri, de_ti = mee.conv_solve()
        mee.fft_type = 1
        de_ri1, de_ti1 = mee.conv_solve()

        try:
            print('de_ri, de_ti norm: ', np.linalg.norm(de_ri - de_ri1), np.linalg.norm(de_ti - de_ti1))
        except:
            print('de_ri, de_ti norm: ', torch.linalg.norm(de_ri - de_ri1),  torch.linalg.norm(de_ti - de_ti1))

    return


if __name__ == '__main__':
    t0 = time.time()

    dtype = 0
    device = 0

    print('NumpyMeent')
    compare_conv_mat_method(0, type_complex=dtype, device=device)
    print('JaxMeent')
    compare_conv_mat_method(1, type_complex=dtype, device=device)
    print('TorchMeent')
    compare_conv_mat_method(2, type_complex=dtype, device=device)
