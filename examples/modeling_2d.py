import torch
import numpy as np
import meent
# from meent.on_numpy.modeler.modeling import ModelingNumpy as Modeling
from meent.on_jax.modeler.modeling import ModelingJax as Modeling
from meent.on_torch.modeler.modeling import ModelingTorch as Modeling
from meent.on_torch.optimizer.loss import LossDeflector

mode = 2

period = [1000, 1000]
thickness = torch.tensor([300.])
layer_base = 1.

length1 = 100
length2 = 40
length3 = 50
length4 = 100


obj1 = [[0, 30], [length1, length2], 9.]
obj2 = [[0, 50], [length3, length4], 9.]

obj_list = [obj1, obj2, ]

layer_info_list = [[period, layer_base, obj_list]]
ucell_info_list = Modeling().draw(layer_info_list)

mee = meent.call_mee(backend=mode, grating_type=2, fft_type=2, ucell_info_list=ucell_info_list, thickness=thickness)
de_ri, de_ti = mee.conv_solve()

print(de_ti)

ucell = np.array([
    [
        [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, ],
        [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, ],
        [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, ],
        [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, ],
        [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, ],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, ],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, ],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, ],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, ],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, ],
    ],
])*2 + 1.

period = [1000, 1000]
thickness = torch.tensor([1120., 400, 300])
# ucell = np.array(
#     [
#         [
#             [3.1, 1.1, 1.2, 1.6, 3.1],
#             [3.5, 1.4, 1.1, 1.2, 3.6],
#         ],
#         [
#             [3.5, 1.2, 1.5, 1.2, 3.3],
#             [3.1, 1.5, 1.5, 1.4, 3.1],
#         ],
#         [
#             [3.5, 1.2, 1.5, 1.2, 3.3],
#             [3.1, 1.5, 1.5, 1.4, 3.1],
#         ],
#     ]
# )

ucell = torch.tensor(ucell)
mee = meent.call_mee(backend=mode, period=period,
                     pol=1, fourier_order=2, grating_type=2, fft_type=0, ucell=ucell, ucell_info_list=ucell_info_list, thickness=thickness)

de_ri, de_ti = mee.conv_solve()

pois = ['ucell', 'thickness']

forward = mee.conv_solve
loss_fn = LossDeflector(x_order=0, y_order=0)

# case 1: Gradient
grad = mee.grad(pois, forward, loss_fn)
print(1, grad['ucell'])
print(de_ti)

