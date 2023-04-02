import os
import pathlib
import sys
from copy import deepcopy

import jax

from meent.on_jax.emsolver.convolution_matrix import read_material_table
# TODO: update or delete

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))

import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ''

os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=6

# os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
# os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=40'


import numpy as np
import jax.numpy as jnp

from meent.main import call_mee
from meent.on_jax.emsolver.transfer_method import *


# common
# grating_type = 1  # 0: 1D, 1: 1D conical, 2:2D.
pol = 1  # 0: TE, 1: TM

n_I = 1  # n_incidence
n_II = 1  # n_transmission

theta = 0 * np.pi / 180
phi = 0 * np.pi / 180
psi = 0 * np.pi / 180 if pol else 90 * np.pi / 180

wavelength = 900

thickness = [500]
ucell_materials = [1, 3.48]

mode_options = {0: 'numpy', 1: 'JAX', 2: 'Torch', }
mat_list = [1, 3.48]
fourier_order = 40  # 15 20 25 30

grating_type = 0

period = [1000]
perturbation = 1E-10

device = 0
dtype = 0

if device == 0:
    device = 'cpu'
    jax.config.update('jax_platform_name', device)
else:
    device = 'gpu'
    jax.config.update('jax_platform_name', device)

if dtype == 0:
    jax.config.update('jax_enable_x64', True)
    type_complex = jnp.complex128
else:
    type_complex = jnp.complex64


ucell = np.array([
    [
        [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, ]*10,
    ],
], dtype=np.float64)

solver = call_mee(backend=1, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi,
                  fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell,
                  ucell_materials=ucell_materials,
                  thickness=thickness, device=device, type_complex=type_complex, )

mat_table = read_material_table()

mat_wl = mat_table['P_SI'][:, 0]
mat_permittivity_real = mat_table['P_SI'][:, 1]
mat_permittivity_imag = mat_table['P_SI'][:, 2]

wavelength_list = jnp.arange(500, 1000, 10)
num_device = jax.device_count()

len_wavelength_list = len(wavelength_list)
counter = len_wavelength_list // num_device

if len_wavelength_list % num_device != 0:
    counter += 1

mat_pmtvy_interp = jnp.interp(wavelength_list, mat_wl, mat_permittivity_real)


def loop_wavelength(wavelength1, permittivity1):
    ucell1 = deepcopy(solver.ucell)
    ucell1 *= permittivity1
    ucell1 += 1.

    solver.wavelength = wavelength1
    de_ri, de_ti = solver.conv_solve_spectrum(ucell1)
    return de_ri, de_ti


def generate_spectrum_pmap():

    spectrum_ri_pmap = np.zeros(wavelength_list.shape)
    spectrum_ti_pmap = np.zeros(wavelength_list.shape)
    for i, wavelength in enumerate(range(counter)):

        b = i * num_device
        de_ri_pmap, de_ti_pmap = jax.pmap(loop_wavelength)(wavelength_list[b:b+num_device], mat_pmtvy_interp[b:b+num_device])

        spectrum_ri_pmap[b:b+num_device] = de_ri_pmap.sum(axis=(1))
        spectrum_ti_pmap[b:b+num_device] = de_ti_pmap.sum(axis=(1))

    return spectrum_ri_pmap, spectrum_ti_pmap


def generate_spectrum():
    spectrum_ri_single = np.zeros(wavelength_list.shape)
    spectrum_ti_single = np.zeros(wavelength_list.shape)

    for i, wavelength in enumerate(wavelength_list):
        de_ri, de_ti = loop_wavelength(wavelength, mat_pmtvy_interp[i])

        spectrum_ri_single[i] = de_ri.sum()
        spectrum_ti_single[i] = de_ti.sum()

    return spectrum_ri_single, spectrum_ti_single

import matplotlib.pyplot as plt

t0 = time.time()
spectrum_ri_single, spectrum_ti_single = generate_spectrum()
print('jax loop 1st:', time.time() - t0)
plt.plot(wavelength_list, spectrum_ri_single)
plt.plot(wavelength_list, spectrum_ti_single)
plt.show()

t0 = time.time()
spectrum_ri_single, spectrum_ti_single = generate_spectrum()
print('jax loop 2nd:', time.time() - t0)
plt.plot(wavelength_list, spectrum_ri_single)
plt.plot(wavelength_list, spectrum_ti_single)
plt.show()

t0 = time.time()
spectrum_ri_single, spectrum_ti_single = generate_spectrum()
print('jax loop 3rd:', time.time() - t0)
plt.plot(wavelength_list, spectrum_ri_single)
plt.plot(wavelength_list, spectrum_ti_single)
plt.show()

t0 = time.time()
spectrum_ri_pmap, spectrum_ti_pmap = generate_spectrum_pmap()
print('jax parallel 1st:', time.time() - t0)
plt.plot(wavelength_list, spectrum_ri_pmap)
plt.plot(wavelength_list, spectrum_ti_pmap)
plt.show()

t0 = time.time()
spectrum_ri_pmap, spectrum_ti_pmap = generate_spectrum_pmap()
print('jax parallel 2nd:', time.time() - t0)
plt.plot(wavelength_list, spectrum_ri_pmap)
plt.plot(wavelength_list, spectrum_ti_pmap)
plt.show()

t0 = time.time()
spectrum_ri_pmap, spectrum_ti_pmap = generate_spectrum_pmap()
print('jax parallel 3rd:', time.time() - t0)
plt.plot(wavelength_list, spectrum_ri_pmap)
plt.plot(wavelength_list, spectrum_ti_pmap)
plt.show()


# print('difference de_ri:', np.linalg.norm(spectrum_ri_pmap - spectrum_ri_single))
# print('difference de_ti:', np.linalg.norm(spectrum_ti_pmap - spectrum_ti_single))

solver1 = call_mee(backend=0, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi,
                   fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell,
                   ucell_materials=ucell_materials,
                   thickness=thickness, device=device, type_complex=type_complex, )


from meent.on_numpy.emsolver.convolution_matrix import to_conv_mat_continuous, to_conv_mat_discrete
t0 = time.time()


spectrum_ri = np.zeros(wavelength_list.shape)
spectrum_ti = np.zeros(wavelength_list.shape)

for i, (wl, perm) in enumerate(zip(wavelength_list, mat_pmtvy_interp)):
    ucell1 = deepcopy(ucell)
    ucell1 *= perm
    ucell1 += 1.

    E_conv_all, o_E_conv_all = to_conv_mat_discrete(ucell1, *fourier_order, type_complex=type_complex,)
    de_ri, de_ti, _,_,_ = solver1.solve(wl, E_conv_all, o_E_conv_all)

    spectrum_ri[i] = de_ri.sum()
    spectrum_ti[i] = de_ti.sum()


print('Numpy loop 1st:', time.time() - t0)
plt.plot(wavelength_list, spectrum_ri)
plt.plot(wavelength_list, spectrum_ti)
plt.show()

print('End')
