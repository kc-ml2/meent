import numpy as np

from meent import call_mee


def compare_conv_mat_method(backend, type_complex, device):

    pol = 1  # 0: TE, 1: TM
    n_top = 1.45  # n_incidence
    n_bot = 1  # n_transmission

    theta = 0
    phi = 0

    wavelength = 900
    fto = 80

    ucell = [1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1,
             1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1]
    ucell = np.array(ucell).reshape((1, 1, -1))
    ucell = (ucell + 1) * 1.5 + 1

    thickness = [325]
    period = [abs(wavelength / np.sin(60 / 180 * np.pi))]
    mee = call_mee(backend, pol=pol, n_top=n_top, n_bot=n_bot, theta=theta, phi=phi,
                   fto=fto, wavelength=wavelength, period=period, ucell=ucell,
                   thickness=thickness, device=device, type_complex=type_complex, )

    mee.fourier_type = 0
    mee.enhanced_dfs = False
    de_ri_dfs, de_ti_dfs = mee.conv_solve()

    mee.enhanced_dfs = True
    de_ri_efs, de_ti_efs = mee.conv_solve()

    mee.fourier_type = 1
    de_ri_cfs, de_ti_cfs = mee.conv_solve()

    a = np.linalg.norm(de_ri_dfs - de_ri_efs)
    b = np.linalg.norm(de_ti_dfs - de_ti_efs)
    c = np.linalg.norm(de_ri_dfs - de_ri_cfs)
    d = np.linalg.norm(de_ti_dfs - de_ti_cfs)
    e = np.linalg.norm(de_ri_efs - de_ri_cfs)
    f = np.linalg.norm(de_ti_efs - de_ti_cfs)

    print('DFS-EFS ', a, b)
    print('DFS-CFS ', c, d)
    print('EFS-CFS ', e, f)


if __name__ == '__main__':
    dtype = 0
    device = 0

    print('NumpyMeent')
    compare_conv_mat_method(0, type_complex=dtype, device=device)
    print('JaxMeent')
    compare_conv_mat_method(1, type_complex=dtype, device=device)
    print('TorchMeent')
    compare_conv_mat_method(2, type_complex=dtype, device=device)
