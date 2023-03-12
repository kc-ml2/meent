import torch

import meent.testcase
from meent.on_torch.optimizer.loss import LossDeflector
from meent.on_torch.optimizer.optimizer import OptimizerTorch


mode = 2
dtype = 0
device = 0
grating_type = 2

conditions = meent.testcase.load_setting(mode, dtype, device, grating_type)
mee = OptimizerTorch(**conditions)

pois = ['ucell', 'thickness']

forward = mee.conv_solve
loss_fn = LossDeflector(x_order=0, y_order=0)

# case 1: Gradient
grad = mee.grad(pois, forward, loss_fn)
print(1, grad)

# case 2: SGD
opt_torch = torch.optim.SGD
opt_options = {'lr': 1E-2,
               'momentum': 0.9,
               }

mee.fit(pois, forward, loss_fn, opt_torch, opt_options)
