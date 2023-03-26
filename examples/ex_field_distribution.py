import time
import numpy as np

from meent.main import call_mee

# common
grating_type = 2  # 0: 1D, 1: 1D conical, 2:2D.
pol = 1  # 0: TE, 1: TM

n_I = 1  # n_incidence
n_II = 1  # n_transmission

theta = 0 * np.pi / 180
phi = 0 * np.pi / 180
psi = 0 * np.pi / 180 if pol else 90 * np.pi / 180

wavelength = 900

thickness = [500]
ucell_materials = ['p_si__real', 3.48]

mode_options = {0: 'numpy', 1: 'JAX', 2: 'Torch', }

if grating_type in (0, 1):
    period = [700]
    fourier_order = 20

    ucell = np.array(
        [
            [
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
            ],
            [
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
            ],
        ]
    )
else:
    period = [700, 700]
    fourier_order = 9

    ucell = np.array(
        [
            [
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
            ],
            [
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ],
            ],
        ]
    )

from meent.on_numpy.modeler.modeling import ModelingNumpy

modeler = ModelingNumpy()
ucell = modeler.put_refractive_index_in_ucell(ucell, ucell_materials, wavelength)

for i in range(3):
    AA = call_mee(backend=i, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
                  fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell,
                  ucell_materials=ucell_materials,
                  thickness=thickness, )

    t0 = time.time()
    de_ri, de_ti = AA.conv_solve()
    print(f'run_cell: ', time.time() - t0)


    resolution = (50, 50, 50)
    t0 = time.time()
    field_cell = AA.calculate_field(resolution=resolution, plot=True)
    print(f'cal_field: ', time.time() - t0)
