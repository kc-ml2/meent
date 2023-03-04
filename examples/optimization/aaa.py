import time
from functools import partial

import numpy as np
import optax
import jax
import jax.numpy as jnp


import meent
from meent.on_jax.optimizer.optimizer import OptimizerJax
from meent.on_jax.optimizer.loss import LossDeflector


mode = 1
dtype = 0
device = 0
grating_type = 2

conditions = meent.testcase.load_setting(mode, dtype, device, grating_type)
solver = OptimizerJax(**conditions)

initial_params = {
    'ucell': solver.ucell,
    'thickness': solver.thickness,
}

forward = solver.conv_solve
optimizer = optax.adam(learning_rate=1e-2)


def loss(value) -> jnp.ndarray:
    de_ri, de_ti = value
    loss_value = optax.sigmoid_binary_cross_entropy(de_ti[5, 5], 1).sum()
    return loss_value


# loss_fn = loss
loss_fn = LossDeflector(x_order=0, y_order=0)


solver.grad(initial_params, forward, loss_fn)
solver.fit(initial_params, forward, loss_fn, optimizer)
print(1)
