import numpy as np
# import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
import time

from meent.rcwa import call_solver


n_I = 1
n_si = 3.48
n_air = 1
n_II = 1
theta = 1E-10
phi = 1E-10

fourier_order = 2

period = [700, 700]
wls = np.array([900.])
pol = 1
psi = 0 if pol else 90  # in degree, notation from Moharam paper

thickness = [1120]

pattern = np.array([n_si, n_si, n_si, n_air, n_air, n_air, n_air, n_air, n_air, n_air, ])

N = len(pattern)
dx = period[0]/N
grid = np.arange(1, N+1)*dx

textures = [n_I, [grid, pattern], n_II]

profile = np.array([[0, *thickness, 0], [1, 2, 3]])
grating_type = 2

pattern = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, ])
ucell_materials = [n_air, n_si]

ucell = np.array([[pattern]])

from meent.on_numpy.convolution_matrix import cell_compression

cell_comp, x, y = cell_compression(ucell[0])

ucell = np.zeros((1, 10, 10))

base = np.meshgrid(0, np.arange(0, 10, 1), np.arange(0, 10, 1))

obj1 = np.meshgrid(0, np.arange(0, 10, 1), np.arange(3, 5, 1))

obj_list = [base, obj1]
mat_list = [1, 'si']

from meent.on_numpy.convolution_matrix import put_permittivity_in_ucell_object, \
    read_material_table, put_permittivity_in_ucell, to_conv_mat

# a = put_permittivity_in_ucell_new(ucell, mat_list, obj_list, None, wls)

# --- Run ---

solver = call_solver(mode=0, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
                 fourier_order=fourier_order, wls=wls, period=period, ucell=ucell, ucell_materials=ucell_materials,
                 thickness=thickness)
t0 = time.time()
# de_ri, de_ti = meent_t.run_ucell()

mat_table = read_material_table()

solver.ucell = put_permittivity_in_ucell_object(ucell, mat_list, obj_list, mat_table, wls)

e_conv_all = to_conv_mat(solver.ucell, solver.fourier_order)
o_e_conv_all = to_conv_mat(1 / solver.ucell, solver.fourier_order)

de_ri, de_ti = solver.solve(solver.wavelength, e_conv_all, o_e_conv_all)

print(de_ri)
