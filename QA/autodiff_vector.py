import meent


def run_jax():
    print('RUN JAXMeent')
    import jax
    import optax
    import jax.numpy as jnp

    backend = 1

    period = [1000., 1000.]
    thickness = ([300.])
    wavelength = 900

    input_length1 = jnp.array([160], dtype=jnp.float64)
    input_length2 = jnp.array([100], dtype=jnp.float64)
    input_length3 = jnp.array([30], dtype=jnp.float64)
    input_length4 = jnp.array([20], dtype=jnp.float64)

    fto = [5, 5]

    mee = meent.call_mee(backend=backend, fto=fto, wavelength=wavelength, thickness=thickness, period=period,
                         device=0, type_complex=0)

    opt = optax.sgd(learning_rate=1E5, momentum=0)

    def forward(param_list):
        [length1, length2, length3, length4] = param_list
        ucell = [
            [3 - 1j, [
                ['rectangle', 0 + 1000, 410 + 1000, length1, 80, 4, 0, 0, 0],  # obj 1
                ['ellipse', 0 + 1000, -10 + 1000, length2, 80, 4, 1, 20, 20],  # obj 2
                ['rectangle', 120 + 1000, 500 + 1000, length3, 160, 4 + 0.3j, 1.1, 5, 5],  # obj 3
                ['ellipse', -400 + 1000, -700 + 1000, length4, 160, 4, 0.4, 20, 20],  # obj 4
            ], ],
        ]
        mee.ucell = ucell

        res = mee.conv_solve().res
        de_ti = res.de_ti

        cy, cx = de_ti.shape[0] // 2,  de_ti.shape[1] // 2
        loss = -de_ti[cy, cx + 1]

        return loss

    pois = [input_length1, input_length2, input_length3, input_length4]
    opt_state = opt.init(pois)

    for i in range(10):
        print('Parameters: ', [p.item() for p in pois])

        input_length1, input_length2, input_length3, input_length4 = pois

        dx = 1E-5
        loss_a = forward([input_length1 + dx, input_length2, input_length3, input_length4])
        loss_b = forward([input_length1 - dx, input_length2, input_length3, input_length4])
        grad1 = (loss_a - loss_b) / (2 * dx)

        loss_a = forward([input_length1, input_length2 + dx, input_length3, input_length4])
        loss_b = forward([input_length1, input_length2 - dx, input_length3, input_length4])
        grad2 = (loss_a - loss_b) / (2 * dx)

        loss_a = forward([input_length1, input_length2, input_length3 + dx, input_length4])
        loss_b = forward([input_length1, input_length2, input_length3 - dx, input_length4])
        grad3 = (loss_a - loss_b) / (2 * dx)

        loss_a = forward([input_length1, input_length2, input_length3, input_length4 + dx])
        loss_b = forward([input_length1, input_length2, input_length3, input_length4 - dx])
        grad4 = (loss_a - loss_b) / (2 * dx)

        print('grad_nume: ', grad1.item(), grad2.item(), grad3.item(), grad4.item())

        # grad = jax.grad(forward)(pois)
        loss, grad = jax.value_and_grad(forward)(pois)
        updates, opt_state = opt.update(grad, opt_state, pois)

        pois = optax.apply_updates(pois, updates)
        print('grad_auto: ', *[g.item() for g in grad])
        print('Loss:', loss)


def run_torch():
    print('RUN TorchMeent')
    import torch
    backend = 2

    period = [1000., 1000.]
    thickness = torch.tensor([300.])
    wavelength = 900

    input_length1 = 160
    input_length2 = 100
    input_length3 = 30
    input_length4 = 20

    fto = [5, 5]

    # layer_base = torch.tensor(n_index_base)
    input_length1 = torch.tensor([input_length1], dtype=torch.float64, requires_grad=True)
    input_length2 = torch.tensor([input_length2], dtype=torch.float64, requires_grad=True)
    input_length3 = torch.tensor([input_length3], dtype=torch.float64, requires_grad=True)
    input_length4 = torch.tensor([input_length4], dtype=torch.float64, requires_grad=True)

    mee = meent.call_mee(backend=backend, fto=fto, wavelength=wavelength, thickness=thickness, period=period,
                         device=0, type_complex=0)

    opt = torch.optim.SGD([input_length1, input_length2, input_length3, input_length4], lr=1E5, momentum=0)

    def forward(length1, length2, length3, length4):

        ucell = [
            [3 - 1j, [
                 ['rectangle', 0+1000, 410+1000, length1, 80, 4, 0, 0, 0],  # obj 1
                 ['ellipse', 0+1000, -10+1000, length2, 80, 4, 1, 20, 20],  # obj 2
                 ['rectangle', 120+1000, 500+1000, length3, 160, 4+0.3j, 1.1, 5, 5],  # obj 3
                 ['ellipse', -400+1000, -700+1000, length4, 160, 4, 0.4, 20, 20],  # obj 4
             ], ],
        ]
        mee.ucell = ucell

        res = mee.conv_solve().res
        de_ti = res.de_ti

        cy, cx = de_ti.shape[0] // 2,  de_ti.shape[1] // 2
        loss = -de_ti[cy, cx + 1]

        return loss

    for i in range(10):
        print('Parameters: ', input_length1.detach().numpy(), input_length2.detach().numpy(),
              input_length3.detach().numpy(), input_length4.detach().numpy())
        dx = 1E-5
        loss_a = forward(input_length1 + dx, input_length2, input_length3, input_length4)
        loss_b = forward(input_length1 - dx, input_length2, input_length3, input_length4)
        grad1 = (loss_a - loss_b) / (2 * dx)

        loss_a = forward(input_length1, input_length2 + dx, input_length3, input_length4)
        loss_b = forward(input_length1, input_length2 - dx, input_length3, input_length4)
        grad2 = (loss_a - loss_b) / (2 * dx)

        loss_a = forward(input_length1, input_length2, input_length3 + dx, input_length4)
        loss_b = forward(input_length1, input_length2, input_length3 - dx, input_length4)
        grad3 = (loss_a - loss_b) / (2 * dx)

        loss_a = forward(input_length1, input_length2, input_length3, input_length4 + dx)
        loss_b = forward(input_length1, input_length2, input_length3, input_length4 - dx)
        grad4 = (loss_a - loss_b) / (2 * dx)

        print('grad_nume: ', grad1.item(), grad2.item(), grad3.item(), grad4.item())

        loss = forward(input_length1, input_length2, input_length3, input_length4)
        loss.backward()
        print('grad_auto: ', input_length1.grad.numpy()[0], input_length2.grad.numpy()[0], input_length3.grad.numpy()[0],
              input_length4.grad.numpy()[0])

        opt.step()
        opt.zero_grad()
        print('Loss:', loss)


if __name__ == '__main__':
    run_jax()
    run_torch()
