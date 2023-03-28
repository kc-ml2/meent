import jax
import optax

import jax.numpy as jnp

import meent

from meent.on_jax.optimizer.loss import LossDeflector


backend = 1
dtype = 0
device = 0
grating_type = 2


pol = 0  # 0: TE, 1: TM

n_I = 1  # n_incidence
n_II = 1  # n_transmission

theta = 20 * jnp.pi / 180
phi = 50 * jnp.pi / 180

wavelength = 900

thickness = jnp.array([550.])
period = [1000., 300.]

fourier_order = [4, 2]

type_complex = jnp.complex128

ucell_1d_s = jnp.array([
    [
        [0, 1, 0, 1, 1.1, 0, 1, 0, 1, 1, ],
    ],
], dtype=jnp.float64) * 4. + 1.  # refractive index

ucell_2d_s = jnp.array([
    [
        [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, ],
        [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, ],
    ],
]) * 4 + 1.  # refractive index

mee = meent.call_mee(backend=backend, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell_2d_s, thickness=thickness, type_complex=type_complex, fft_type=0, improve_dft=True)
pois = ['ucell', 'thickness']

forward = mee.conv_solve
loss_fn = LossDeflector(x_order=0, y_order=0)

# case 1: Gradient
grad = mee.grad(pois, forward, loss_fn)
print('Gradient: ', grad)

# case 2: SGD
optimizer = optax.sgd(learning_rate=1e-2)
res = mee.fit(pois, forward, loss_fn, optimizer, iteration=100)
print('Fitting with SGD', res)

optimizer = optax.adam(learning_rate=1e-2)
res = mee.fit(pois, forward, loss_fn, optimizer, iteration=100)
print('Fitting with ADAM', res)
