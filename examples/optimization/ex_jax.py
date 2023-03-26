import optax

import meent
from meent.on_jax.optimizer.loss import LossDeflector
from meent.on_jax.optimizer.optimizer import OptimizerJax


backend = 1
dtype = 0
device = 0
grating_type = 2

conditions = meent.testcase.load_setting(backend, dtype, device, grating_type)
mee = OptimizerJax(**conditions)

pois = ['ucell', 'thickness']

forward = mee.conv_solve
loss_fn = LossDeflector(x_order=0, y_order=0)

# case 1: Gradient
grad = mee.grad(pois, forward, loss_fn)
print(1, grad)

# case 2: SGD
optimizer = optax.sgd(learning_rate=1e-2)
mee.fit(pois, forward, loss_fn, optimizer)
print(3, mee.thickness*1E5)
