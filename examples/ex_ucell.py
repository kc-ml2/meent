import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'

# os.environ["MKL_NUM_THREADS"] = "8"  # export MKL_NUM_THREADS=6
# os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
# os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6

import time
import numpy as np

import meent


# common
pol = 0  # 0: TE, 1: TM

n_I = 1  # n_incidence
n_II = 1  # n_transmission

theta = 20 * np.pi / 180
phi = 50 * np.pi / 180
psi = 0 if pol else 90 * np.pi / 180

wavelength = 900

thickness = [500]
ucell_materials = [1, 'p_si__real']
period = [1000, 1000]

fourier_order = [20, 4]
mode_options = {0: 'numpy', 1: 'JAX', 2: 'Torch', }
n_iter_de = 2
n_iter_field = 1


def run_test(grating_type, backend, dtype, device):
    ucell = load_ucell(grating_type)

    mee = meent.call_mee(backend=backend, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi,
                         fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell,
                         ucell_materials=ucell_materials,
                         thickness=thickness, device=device, type_complex=dtype, fft_type=0, improve_dft=True)
    # mee.fft_type = 1
    # mee.device = 1
    # mee.type_complex = 1
    resolution = (20, 20, 20)

    for i in range(n_iter_de):
        t0 = time.time()
        de_ri, de_ti = mee.conv_solve()
        print(f'run_cell: {i}: ', time.time() - t0)
        try:
            print('de_ri: ', de_ri.numpy()[de_ri.shape[0]//2])
        except:
            print('de_ri: ', de_ri[de_ri.shape[0]//2])
    for i in range(n_iter_field):
        t0 = time.time()
        field_cell = mee.calculate_field(res_x=resolution[0], res_y=resolution[1], res_z=resolution[2])
        print(f'cal_field: {i}', time.time() - t0)
        # mee.cal_field(field_cell)

    for i in range(n_iter_field):
        t0 = time.time()
        de_ri, de_ti, field_cell = mee.conv_solve_field(res_x=resolution[0], res_y=resolution[1], res_z=resolution[2])
        print(f'cal_field: {i}', time.time() - t0)
        # mee.cal_field(field_cell)


def run_loop(a, b, c, d):
    for grating_type in a:
        for bd in b:
            for dtype in c:
                for device in d:
                    print(f'grating:{grating_type}, backend:{bd}, dtype:{dtype}, dev:{device}')
                    run_test(grating_type, bd, dtype, device)
                    print('\n')


def load_ucell(grating_type):
    if grating_type in [0, 1]:

        ucell = np.array([

            [
                [
                    0, 1, 0, 1, 1, 0, 1, 0, 1, 1,
                ],
            ],
        ])*4 + 1
    else:

        ucell = np.array([
            [
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
            ],
        ])

        ucell = np.array([
            [
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, ],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, ],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, ],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, ],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, ],
            ],
        ]) * 4 + 1

    return ucell


if __name__ == '__main__':
    # run_loop([0], [1], [0], [0])
    # run_loop([1], [0,2], [0], [0])
    # run_loop([0], [1], [0], [0])
    run_loop([0,1,2], [0,1,2], [0], [0])
    run_loop([0,1,2], [0,1,2], [1], [0])
