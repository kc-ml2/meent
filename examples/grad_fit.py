import time

import jax
import optax
import numpy as np
import jax.numpy as jnp

import torch

import meent
from meent.on_torch.optimizer.loss import LossDeflector


backend = 1  # JAX

# common
pol = 0  # 0: TE, 1: TM

n_I = 1  # n_incidence
n_II = 1  # n_transmission

theta = 0 * jnp.pi / 180  # angle of incidence
phi = 0 * jnp.pi / 180  # angle of rotation

wavelength = 900

period = [1000.]  # length of the unit cell. Here it's 1D.

fourier_order = [10]

type_complex = jnp.complex128

grating_type = 0  # grating type: 0 for 1D grating without rotation (phi == 0)
thickness = [500., 1000.]  # thickness of each layer, from top to bottom.

ucell_1d_m = np.array([
    [[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, ]],
    [[1, 1, 1, 1, 0, 1, 1, 1, 1, 1, ]],
    ]) * 4. + 1.  # refractive index

mee = meent.call_mee(backend=backend, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell_1d_m, thickness=thickness, type_complex=type_complex, fft_type=0, improve_dft=True)

pois = ['ucell', 'thickness']
forward = mee.conv_solve
loss_fn = LossDeflector(x_order=1, y_order=0)

# case 1: Gradient
grad = mee.grad(pois, forward, loss_fn)

print('ucell gradient:')
print(grad['ucell'])
print('thickness gradient:')
print(grad['thickness'])

thickness = [500., 1000.]  # thickness of each layer, from top to bottom.

ucell_1d_m = np.array([
    [[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, ]],
    [[1, 1, 1, 1, 0, 1, 1, 1, 1, 1, ]],
    ]) * 4. + 1.  # refractive index

mee = meent.call_mee(backend=backend, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell_1d_m, thickness=thickness, type_complex=type_complex, fft_type=0, improve_dft=True)

pois = ['ucell', 'thickness']
forward = mee.conv_solve
loss_fn = LossDeflector(x_order=1, y_order=0)

# case 2: SGD
optimizer = optax.sgd(learning_rate=1e-2, momentum=0.9)
t0 = time.time()
res = mee.fit(pois, forward, loss_fn, optimizer, iteration=10000)
print('Time JAX', time.time() - t0)

print('ucell final:')
print(res['ucell'])
print('thickness final:')
print(res['thickness'])

backend = 2  # Torch

pol = 0  # 0: TE, 1: TM

n_I = 1  # n_incidence
n_II = 1  # n_transmission

theta = 0 * torch.pi / 180  # angle of incidence
phi = 0 * torch.pi / 180  # angle of rotation

wavelength = 900

thickness = torch.tensor([500., 1000.])  # thickness of each layer, from top to bottom.
period = torch.tensor([1000.])  # length of the unit cell. Here it's 1D.

fourier_order = [10]

type_complex = torch.complex128
device = torch.device('cpu')

grating_type = 0  # grating type: 0 for 1D grating without rotation (phi == 0)

mee = meent.call_mee(backend=backend, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell_1d_m, thickness=thickness, type_complex=type_complex, device=device, fft_type=0, improve_dft=True)

mee.ucell.requires_grad = True
mee.thickness.requires_grad = True

de_ri, de_ti = mee.conv_solve()
loss = de_ti[de_ti.shape[0] // 2 + 1]

loss.backward()
print('ucell gradient:')
print(mee.ucell.grad)
print('thickness gradient:')
print(mee.thickness.grad)

thickness = [500., 1000.]  # thickness of each layer, from top to bottom.

ucell_1d_m = np.array([
    [[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, ]],
    [[1, 1, 1, 1, 0, 1, 1, 1, 1, 1, ]],
    ]) * 4. + 1.  # refractive index

mee = meent.call_mee(backend=backend, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell_1d_m, thickness=thickness, type_complex=type_complex, device=device, fft_type=0, improve_dft=True)

pois = ['ucell', 'thickness']

forward = mee.conv_solve
loss_fn = LossDeflector(1, 0)

opt_torch = torch.optim.SGD
opt_options = {'lr': 1E-2,
               'momentum': 0.9,
               }
t0 = time.time()
res = mee.fit(pois, forward, loss_fn, opt_torch, opt_options, iteration=10000)
print('Time Torch: ', time.time() - t0)
print('ucell final:')
print(res[0])
print('thickness final:')
print(res[1])


# import optax
#
# import meent
# from meent.on_jax.optimizer.loss import LossDeflector
# from meent.on_jax.optimizer.optimizer import OptimizerJax
#
# import jax
# import optax
#
# import jax.numpy as jnp
#
# import meent
# from meent.on_jax.optimizer.loss import LossDeflector
#
#
# mode = 1
# dtype = 0
# device = 0
# grating_type = 2
#
# # conditions = meent.testcase.load_setting(mode, dtype, device, grating_type)
#
# backend = 1  # JAX
#
# # common
# pol = 0  # 0: TE, 1: TM
#
# n_I = 1  # n_incidence
# n_II = 1  # n_transmission
#
# theta = 0 * jnp.pi / 180  # angle of incidence
# phi = 0 * jnp.pi / 180  # angle of rotation
#
# wavelength = 900
#
# thickness = [500., 1000.]  # thickness of each layer, from top to bottom.
# period = [1000.]  # length of the unit cell. Here it's 1D.
#
# fourier_order = [10]
# ucell_1d_m = jnp.array([
#     [[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, ]],
#     [[1, 1, 1, 1, 0, 1, 1, 1, 1, 1, ]],
#     ]) * 4. + 1.  # refractive index
#
#
# type_complex = jnp.complex128
#
# grating_type = 0  # grating type: 0 for 1D grating without rotation (phi == 0)
# mee = meent.call_mee(backend=backend, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell_1d_m, thickness=thickness, type_complex=type_complex, fft_type=0, improve_dft=True)
#
# pois = ['ucell', 'thickness']
# forward = mee.conv_solve
# loss_fn = LossDeflector(x_order=1, y_order=0)
#
# # case 1: Gradient
# grad = mee.grad(pois, forward, loss_fn)
#
# print('ucell gradient:')
# print(grad['ucell'])
# print('thickness gradient:')
# print(grad['thickness'])
# pois = ['ucell', 'thickness']
#
# forward = mee.conv_solve
# loss_fn = LossDeflector(x_order=0, y_order=0)
#
# # case 1: Gradient
# grad = mee.grad(pois, forward, loss_fn)
# print(1, grad)
#
# # case 2: SGD
# optimizer = optax.sgd(learning_rate=1e-2)
# mee.fit(pois, forward, loss_fn, optimizer)
# print(3, mee.thickness*1E5)
#
# import torch
#
# import meent.testcase
# from meent.on_torch.optimizer.loss import LossDeflector
# from meent.on_torch.optimizer.optimizer import OptimizerTorch
#
#
# mode = 2
# dtype = 0
# device = 0
# grating_type = 2
#
# conditions = meent.testcase.load_setting(mode, dtype, device, grating_type)
# mee = OptimizerTorch(**conditions)
#
# pois = ['ucell', 'thickness']
#
# forward = mee.conv_solve
# loss_fn = LossDeflector(x_order=0, y_order=0)
#
# # case 1: Gradient
# grad = mee.grad(pois, forward, loss_fn)
# print(1, grad)
#
# # case 2: SGD
# opt_torch = torch.optim.SGD
# opt_options = {'lr': 1E-2,
#                'momentum': 0.9,
#                }
#
# mee.fit(pois, forward, loss_fn, opt_torch, opt_options)
