import numpy as np
# import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
import time

from meent.rcwa import call_solver
from meent.on_numpy.convolution_matrix import put_permittivity_in_ucell_object, \
    read_material_table, put_permittivity_in_ucell, to_conv_mat


n_I = 1
n_si = 3.48
n_air = 1
n_II = 1
theta = 1E-10
phi = 1E-10

fourier_order = 2

period = [700, 700]
wavelength = np.array([900])
pol = 1
psi = 0 if pol else 90  # in degree, notation from Moharam paper

thickness = [1120]

pattern = np.array([n_si, n_si, n_si, n_air, n_air, n_air, n_air, n_air, n_air, n_air, ])

N = len(pattern)
dx = period[0] / N
grid = np.arange(1, N + 1) * dx

textures = [n_I, [grid, pattern], n_II]

profile = np.array([[0, *thickness, 0], [1, 2, 3]])
grating_type = 2

mat_list = [1, 'si', 3, 4]

mat_table = read_material_table()

# 1D
ucell_size = (1, 1, 10)

base = np.meshgrid(0, 0, np.arange(0, 10, 1))

obj1 = np.meshgrid(0, 0, np.array([4, 5, 0, 2]))

obj_list = [base, obj1, ]
aa = put_permittivity_in_ucell_object(ucell_size, mat_list, obj_list, mat_table, wavelength)
plt.imshow(abs(aa[0]))
plt.show()

# 2D
ucell_size = (1, 10, 10)

base = np.meshgrid(0, np.arange(0, 10, 1), np.arange(0, 10, 1))

obj1 = np.meshgrid(0, np.arange(0, 10, 1), np.arange(3, 5, 1))
obj2 = np.meshgrid(0, np.arange(0, 10, 1), np.array([0, 1, 2, 8]))
obj3 = np.meshgrid(0, [np.arange(2, 3, 1), ], np.array([4, 5]))

obj_list = [base, obj1, obj2, obj3]
aa = put_permittivity_in_ucell_object(ucell_size, mat_list, obj_list, mat_table, wavelength)
plt.imshow(abs(aa[0]))
plt.show()

# generate UCELL
# pixel-based or object-based

# pixel based;
pattern_pixel = np.array(
    [
        [
            [1, 1, 1, 0, 0, 0, 0, 0, 1, 0, ],
            [1, 1, 1, 0, 0, 0, 0, 0, 1, 0, ],
        ]])
mat_list = [1, 'si__real', 3, 4]

solver = call_solver(mode=0, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
                     fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=pattern_pixel, ucell_materials=mat_list,
                     thickness=thickness)
t0 = time.time()
de_ri, de_ti = solver.run_ucell()
print(de_ri)

# pixel based;
ucell = put_permittivity_in_ucell(pattern_pixel, mat_list, mat_table, wavelength)

e_conv_all = to_conv_mat(ucell, solver.fourier_order)
o_e_conv_all = to_conv_mat(1 / ucell, solver.fourier_order)

de_ri, de_ti = solver.solve(solver.wavelength, e_conv_all, o_e_conv_all)

print(de_ri)

# object based;
# pattern_object = np.zeros((1, 10, 10))
ucell_size = [1, 10, 10]  # Z Y X

base = np.meshgrid(np.arange(ucell_size[0]), np.arange(ucell_size[1]), np.arange(ucell_size[2]))
obj1 = np.meshgrid(0, np.arange(0, 10, 1), np.arange(3, 5, 1))
obj2 = np.meshgrid(0, np.arange(0, 10, 1), np.array([0, 1, 2, 8]))
obj3 = np.meshgrid(0, np.arange(2, 3, 1), np.array([4, 5]))

obj_list = [base, obj1, obj2, obj3]
ucell = put_permittivity_in_ucell_object(ucell_size, mat_list, obj_list, mat_table, wavelength)

e_conv_all = to_conv_mat(ucell, solver.fourier_order)
o_e_conv_all = to_conv_mat(1 / ucell, solver.fourier_order)

de_ri, de_ti = solver.solve(solver.wavelength, e_conv_all, o_e_conv_all)

print(de_ri)
