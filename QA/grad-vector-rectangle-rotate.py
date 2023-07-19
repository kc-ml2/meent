import torch
import matplotlib.pyplot as plt

import meent


torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)

backend = 2

period = [1000., 1000.]
thickness = torch.tensor([300.])
wavelength = 900

length_x = 100
length_y = 300
c1 = [300, 500]

n_index = 4
n_index_base = 2

fourier_order = [5, 5]

layer_base = torch.tensor(n_index_base)

length_x = torch.tensor(length_x, dtype=torch.float64, requires_grad=True)
length_y = torch.tensor(length_y, dtype=torch.float64, requires_grad=True)


angle = torch.tensor((180 + 45)*torch.pi/180, requires_grad=True)

mee = meent.call_mee(backend=backend, grating_type=2, fft_type=2, fourier_order=fourier_order,
                     wavelength=wavelength, thickness=thickness, period=period, device=0, type_complex=0)

opt = torch.optim.SGD([length_x, length_y, angle], lr=1E2, momentum=1)


def forward(length1, length2, angle, c1):

    length1 = length1.type(torch.complex128)
    length2 = length2.type(torch.complex128)

    obj_list = mee.rectangle_rotate(*c1, length1, length2, 5, 5, n_index, angle)

    layer_info_list = [[layer_base, obj_list]]

    mee.draw(layer_info_list)

    de_ri, de_ti = mee.conv_solve()

    center = de_ti.shape[0] // 2
    loss = -de_ti[center + 0, center + 0]

    return loss, obj_list


def plot(c, leng1, leng2, obj_list_out):
    cx, cy = c
    import matplotlib as mpl
    fig, ax = plt.subplots()
    rec = mpl.patches.Rectangle(xy=(cx - leng1.detach()/2, cy - leng2.detach()/2),
                                 width=leng1.detach(), height=leng2.detach(),
                                 angle=angle*180/torch.pi, rotation_point='center', alpha=0.2)

    ax.add_artist(rec)

    for obj in obj_list_out:
        xy = (obj[0][1][0].detach(), obj[0][0][0].detach())
        width = abs(obj[1][1][0].detach() - obj[0][1][0].detach())
        height = abs(obj[1][0][0].detach() - obj[0][0][0].detach())
        rec = mpl.patches.Rectangle(xy=xy, width=width, height=height,
                                    angle=0, rotation_point='center', alpha=0.2, facecolor='r')
        ax.add_artist(rec)

    plt.xlim(0, period[0])
    plt.ylim(0, period[1])

    plt.show()


for i in range(50):

    loss, obj_list_out = forward(length_x, length_y, angle, c1)
    loss.backward()
    print('loss', loss.detach().numpy())

    print('grad_anal',
          length_x.grad.detach().numpy(),
          length_y.grad.detach().numpy(),
          angle.grad.detach().numpy(),
          )


    dx = 1E-6
    loss_a, _ = forward(length_x + dx, length_y, angle, c1)
    loss_b, _ = forward(length_x - dx, length_y, angle, c1)
    grad1 = (loss_a - loss_b) / (2 * dx)

    loss_a, _ = forward(length_x, length_y + dx, angle, c1)
    loss_b, _ = forward(length_x, length_y - dx, angle, c1)
    grad2 = (loss_a - loss_b) / (2 * dx)

    loss_a, _ = forward(length_x, length_y, angle + dx, c1)
    loss_b, _ = forward(length_x, length_y, angle - dx, c1)
    grad5 = (loss_a - loss_b) / (2 * dx)

    print('grad_nume',
          grad1.detach().numpy(),
          grad2.detach().numpy(),
          grad5.detach().numpy(),
          )

    if i % 1 == 0:
        plot(c1, length_x, length_y, obj_list_out)

    angle.grad.data.clamp_(-0.001, 0.001)

    opt.step()
    opt.zero_grad()

    print()

print(length_x, length_y, angle)
