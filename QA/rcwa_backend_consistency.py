import numpy as np

from meent.main import call_mee


grating_type = 2  # 0: 1D, 1: 1D conical, 2:2D.
pol = 0  # 0: TE, 1: TM

n_I = 1  # n_incidence
n_II = 1  # n_transmission

theta = 20 * np.pi / 180
phi = 30 * np.pi / 180
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
    ) * 3.5 + 1
else:
    period = [700, 700]
    fourier_order = [9, 3]

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
    ) * 3.5 + 1

type_complex = 0
resolution = (50, 50, 50)

# Numpy
mee = call_mee(backend=0, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi,
               fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell,
               ucell_materials=ucell_materials,
               thickness=thickness, )

de_ri_numpy, de_ti_numpy = mee.conv_solve()
field_cell_numpy = mee.calculate_field(res_x=resolution[0], res_y=resolution[1], res_z=resolution[2])

# JAX
mee = call_mee(backend=1, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi,
               fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell,
               ucell_materials=ucell_materials, thickness=thickness, type_complex=type_complex)

de_ri_jax, de_ti_jax = mee.conv_solve()
field_cell_jax = mee.calculate_field(res_x=resolution[0], res_y=resolution[1], res_z=resolution[2])

# Torch
mee = call_mee(backend=2, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi,
               fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell,
               ucell_materials=ucell_materials, thickness=thickness, type_complex=type_complex)

de_ri_torch, de_ti_torch = mee.conv_solve()
de_ri_torch, de_ti_torch = de_ri_torch.numpy(), de_ti_torch.numpy()

field_cell_torch = mee.calculate_field(res_x=resolution[0], res_y=resolution[1], res_z=resolution[2])
field_cell_torch = field_cell_torch.numpy()

print('diffraction efficiency, reflectance')
print('normalized norm(numpy - jax): ', np.linalg.norm(de_ri_numpy - de_ri_jax) / de_ri_numpy.size)
print('normalized norm(jax - torch): ', np.linalg.norm(de_ri_jax - de_ri_torch) / de_ri_numpy.size)
print('normalized norm(torch - numpy): ', np.linalg.norm(de_ri_torch - de_ri_numpy) / de_ri_numpy.size)

print('diffraction efficiency, transmittance')
print('normalized norm(numpy - jax): ', np.linalg.norm(de_ti_numpy - de_ti_jax) / de_ti_numpy.size)
print('normalized norm(jax - torch): ', np.linalg.norm(de_ti_jax - de_ti_torch) / de_ti_numpy.size)
print('normalized norm(torch - numpy): ', np.linalg.norm(de_ti_torch - de_ti_numpy) / de_ti_numpy.size)

print('field distribution')
print('normalized norm(numpy - jax): ', np.linalg.norm(field_cell_numpy - field_cell_jax) / field_cell_numpy.size)
print('normalized norm(jax - torch): ', np.linalg.norm(field_cell_jax - field_cell_torch) / field_cell_numpy.size)
print('normalized norm(torch - numpy): ', np.linalg.norm(field_cell_torch - field_cell_numpy) / field_cell_numpy.size)
