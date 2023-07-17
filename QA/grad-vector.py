import torch

import meent

from meent.on_torch.modeler.modeling import ModelingTorch

backend = 2

period = [1000., 1000.]
thickness = torch.tensor([300.])
wavelength = 900

input_length1 = 100
input_length2 = 100
input_length3 = 100
input_length4 = 100
c1 = [500, 500]
c2 = [700, 550]


n_index = 4
n_index_base = 2

fourier_order = [5, 5]

layer_base = torch.tensor(n_index_base)
input_length1 = torch.tensor(input_length1, dtype=torch.float64, requires_grad=True)
input_length2 = torch.tensor(input_length2, dtype=torch.float64, requires_grad=True)
input_length3 = torch.tensor(input_length3, dtype=torch.float64, requires_grad=True)
input_length4 = torch.tensor(input_length4, dtype=torch.float64, requires_grad=True)

mee = meent.call_mee(backend=backend, grating_type=2, fft_type=2, fourier_order=fourier_order,
                     wavelength=wavelength, thickness=thickness, period=period, device=0, type_complex=0)

opt = torch.optim.SGD([input_length1, input_length2, input_length3, input_length4], lr=1E0, momentum=1)


def forward(input_length1, input_length2, input_length3, input_length4, period, thickness, wavelength, c1, c2):

    length1 = input_length1.type(torch.complex128)
    length2 = input_length2.type(torch.complex128)
    length3 = input_length3.type(torch.complex128)
    length4 = input_length4.type(torch.complex128)

    obj1_list = ModelingTorch.rectangle(*c1, length1, length2, n_index)
    obj2_list = ModelingTorch.rectangle(*c2, length3, length4, n_index + 2)

    obj_list = obj1_list + obj2_list

    layer_info_list = [[layer_base, obj_list]]
    ucell_info_list = mee.draw(layer_info_list)

    mee.ucell_info_list = ucell_info_list

    de_ri, de_ti = mee.conv_solve()

    center = de_ti.shape[0] // 2
    loss = -de_ti[center + 0, center + 0]

    return loss, obj_list


for i in range(50):

    loss, obj2_list = forward(input_length1, input_length2, input_length3, input_length4, period, thickness, wavelength, c1, c2)
    print(loss)
    loss.backward()
    try:
        print('grad_reti', input_length1.grad.numpy(), input_length2.grad.numpy(), input_length3.grad.numpy(), input_length4.grad.numpy())
    except:
        print('grad_reti', input_length1.grad, input_length2.grad, input_length3.grad, input_length4.grad)

    dx = 1E-5
    loss_a, _ = forward(input_length1 + dx, input_length2, input_length3, input_length4, period, thickness, wavelength, c1, c2)
    loss_b, _ = forward(input_length1 - dx, input_length2, input_length3, input_length4, period, thickness, wavelength, c1, c2)
    grad1 = (loss_a - loss_b) / (2 * dx)

    loss_a, _ = forward(input_length1, input_length2 + dx, input_length3, input_length4, period, thickness, wavelength, c1, c2)
    loss_b, _ = forward(input_length1, input_length2 - dx, input_length3, input_length4, period, thickness, wavelength, c1, c2)
    grad2 = (loss_a - loss_b) / (2 * dx)

    loss_a, _ = forward(input_length1, input_length2, input_length3 + dx, input_length4, period, thickness, wavelength, c1, c2)
    loss_b, _ = forward(input_length1, input_length2, input_length3 - dx, input_length4, period, thickness, wavelength, c1, c2)
    grad3 = (loss_a - loss_b) / (2 * dx)

    loss_a, _ = forward(input_length1, input_length2, input_length3, input_length4 + dx, period, thickness, wavelength, c1, c2)
    loss_b, _ = forward(input_length1, input_length2, input_length3, input_length4 - dx, period, thickness, wavelength, c1, c2)
    grad4 = (loss_a - loss_b) / (2 * dx)

    print('grad_nume', grad1.detach().numpy(), grad2.detach().numpy(), grad3.detach().numpy(), grad4.detach().numpy())

    input_length1.grad.data.clamp_(-0.1, 0.1)
    input_length2.grad.data.clamp_(-0.1, 0.1)
    input_length3.grad.data.clamp_(-0.1, 0.1)
    input_length4.grad.data.clamp_(-0.1, 0.1)

    opt.step()
    opt.zero_grad()

    print()

print(input_length1, input_length2, input_length3, input_length4)
