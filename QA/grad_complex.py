import time

import jax
import optax
import numpy as np
import jax.numpy as jnp

import torch

import meent
from meent.on_torch.optimizer.loss import LossDeflector

type_complex = 0
device = 0
n_I = 1  # n_incidence
n_II = 1  # n_transmission
theta = 0 * np.pi / 180  # angle of incidence
phi = 0 * np.pi / 180  # angle of rotation
wavelength = 900

pol = 0  # 0: TE, 1: TM

# case 1
iteration = 10000
grating_type = 0  # grating type: 0 for 1D grating without rotation (phi == 0)
fourier_order = [10]
period = [1000]  # length of the unit cell. Here it's 1D.
thickness = [500, 1000]  # thickness of each layer, from top to bottom.
ucell = np.array([
    [[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, ]],
    [[1, 1, 1, 1, 0, 1, 1, 1, 1, 1, ]],
]) * 4 + 2  # refractive index

# case 2
iteration = 500
grating_type = 2  # grating type: 0 for 1D grating without rotation (phi == 0)
fourier_order = [5, 5]
period = [1000, 1000]  # length of the unit cell. Here it's 1D.
thickness = [500]  # thickness of each layer, from top to bottom.
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

# JAX Meent
jmee = meent.call_mee(backend=1, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi,
                      fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell,
                      thickness=thickness, type_complex=type_complex, device=device, fft_type=0, improve_dft=True)

pois = ['ucell',]  # Parameter Of Interests
forward = jmee.conv_solve
loss_fn = LossDeflector(x_order=0, y_order=0)
# TODO: LossDeflector cross-platform?

# case 1: Gradient
grad = jmee.grad(pois, forward, loss_fn)

print('ucell gradient:')
print(grad['ucell'])
# print('thickness gradient:')
# print(grad['thickness'])

optimizer = optax.sgd(learning_rate=1e-2)
t0 = time.time()
res = jmee.fit(pois, forward, loss_fn, optimizer, iteration=iteration)
print('Time JAX', time.time() - t0)

print('ucell final:')
print(res['ucell'])
# print('thickness final:')
# print(res['thickness'])

# Torch Meent
tmee = meent.call_mee(backend=2, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi,
                      fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell,
                      thickness=thickness, type_complex=type_complex, device=device, fft_type=0, improve_dft=True)

forward = tmee.conv_solve
loss_fn = LossDeflector(x_order=0)  # predefined in meent

grad = tmee.grad(pois, forward, loss_fn)
print('ucell gradient:')
print(grad['ucell'])
# print('thickness gradient:')
# print(grad['thickness'])

opt_torch = torch.optim.SGD
opt_options = {'lr': 1E-2}

t0 = time.time()
res = tmee.fit(pois, forward, loss_fn, opt_torch, opt_options, iteration=iteration)
print('Time Torch: ', time.time() - t0)
print('ucell final:')
print(res[0])
# print('thickness final:')
# print(res[1])
