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
n_top = 1  # n_incidence
n_bot = 1  # n_transmission
theta = 0/180 * np.pi  # angle of incidence
phi = 0/180 * np.pi  # angle of rotation
wavelength = 900

pol = 0  # 0: TE, 1: TM
iteration = 20

fto = [5, 5]
period = [1000, 1000]  # length of the unit cell. Here it's 1D.
thickness = [500]  # thickness of each layer, from top to bottom.

ucell = np.array([[[2.58941352 + 0.47745679j, 4.17771602 + 0.88991205j,
                    2.04255624 + 2.23670125j, 2.50478974 + 2.05242759j,
                    3.32747593 + 2.3854387j],
                   [2.80118605 + 0.53053715j, 4.46498861 + 0.10812571j,
                    3.99377545 + 1.0441131j, 3.10728537 + 0.6637353j,
                    4.74697849 + 0.62841253j],
                   [3.80944424 + 2.25899274j, 3.70371553 + 1.32586402j,
                    3.8011133 + 1.49939415j, 3.14797238 + 2.91158289j,
                    4.3085404 + 2.44344691j],
                   [2.22510179 + 2.86017146j, 2.36613053 + 2.82270351j,
                    4.5087168 + 0.2035904j, 3.15559949 + 2.55311298j,
                    4.29394604 + 0.98362617j],
                   [3.31324163 + 2.77590131j, 2.11744834 + 1.65894674j,
                    3.59347907 + 1.28895345j, 3.85713467 + 1.90714056j,
                    2.93805426 + 2.63385392j]]])

# JAX Meent
jmee = meent.call_mee(backend=1, pol=pol, n_top=n_top, n_bot=n_bot, theta=theta, phi=phi,
                      fto=fto, wavelength=wavelength, period=period, ucell=ucell,
                      thickness=thickness, type_complex=type_complex, device=device)

pois = ['ucell', 'thickness']  # Parameter Of Interests
forward = jmee.conv_solve
loss_fn = LossDeflector(x_order=0, y_order=0)

# case 1: Gradient
grad_j = jmee.grad(pois, forward, loss_fn)

print('ucell gradient:')
print(grad_j['ucell'])
print('thickness gradient:')
print(grad_j['thickness'])

optimizer = optax.sgd(learning_rate=1e-2)
t0 = time.time()
res_j = jmee.fit(pois, forward, loss_fn, optimizer, iteration=iteration)
print('Time JAX', time.time() - t0)

print('ucell final:')
print(res_j['ucell'])
print('thickness final:')
print(res_j['thickness'])

# Torch Meent
tmee = meent.call_mee(backend=2, pol=pol, n_top=n_top, n_bot=n_bot, theta=theta, phi=phi,
                      fto=fto, wavelength=wavelength, period=period, ucell=ucell,
                      thickness=thickness, type_complex=type_complex, device=device)

forward = tmee.conv_solve
loss_fn = LossDeflector(x_order=0)  # predefined in meent

grad_t = tmee.grad(pois, forward, loss_fn)
print('ucell gradient:')
print(grad_t['ucell'])
print('thickness gradient:')
print(grad_t['thickness'])

opt_torch = torch.optim.SGD
opt_options = {'lr': 1E-2}

t0 = time.time()
res_t = tmee.fit(pois, forward, loss_fn, opt_torch, opt_options, iteration=iteration)
print('Time Torch: ', time.time() - t0)
print('ucell final:')
print(res_t[0])
print('thickness final:')
print(res_t[1])

print('\n=============Difference between JaxMeent and TorchMeent==============================\n')
print('initial ucell gradient difference', np.linalg.norm(grad_j['ucell'].conj() - grad_t['ucell'].detach().numpy()))
print('initial thickness gradient difference', np.linalg.norm(grad_j['thickness'].conj() - grad_t['thickness'].detach().numpy()))

print('final ucell difference', np.linalg.norm(res_j['ucell'] - res_t[0].detach().numpy()))
print('final thickness difference', np.linalg.norm(res_j['thickness'] - res_t[1].detach().numpy()))

print('End')

# Note that the gradient in JAX is conjugated.
# https://github.com/google/jax/issues/4891
# https://pytorch.org/docs/stable/notes/autograd.html#autograd-for-complex-numbers
