import numpy as np

from copy import deepcopy

from meent import call_mee


def load_setting():
    pol = 1  # 0: TE, 1: TM

    n_I = 1  # n_incidence
    n_II = 1  # n_transmission

    theta = 0 * np.pi / 180
    phi = 0 * np.pi / 180
    psi = 0 * np.pi / 180 if pol else 90 * np.pi / 180

    wavelength = 900

    fourier_order = [2, 2]

    # case 1
    grating_type = 2
    period = [1000, 1000]
    thickness = [1120., 400, 300]

    ucell = np.array(
        [
            [
                [3.1, 1.1, 1.2, 1.6, 3.1],
                [3.5, 1.4, 1.1, 1.2, 3.6],
            ],
            [
                [3.5, 1.2, 1.5, 1.2, 3.3],
                [3.1, 1.5, 1.5, 1.4, 3.1],
            ],
            [
                [3.5, 1.2, 1.5, 1.2, 3.3],
                [3.1, 1.5, 1.5, 1.4, 3.1],
            ],
        ]
    )

    # # case 2
    # grating_type = 2
    # period = [100, 100]
    # thickness = [1120.]
    # ucell = np.array([
    #     [
    #         [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, ],
    #         [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, ],
    #         [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, ],
    #         [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, ],
    #         [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, ],
    #         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, ],
    #         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, ],
    #         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, ],
    #         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, ],
    #         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, ],
    #     ],
    # ]) * 8 + 1.
    #
    # # case 3
    # grating_type = 0  # grating type: 0 for 1D grating without rotation (phi == 0)
    # thickness = [500, 1000]  # thickness of each layer, from top to bottom.
    # ucell = np.array([
    #     [[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, ]],
    #     [[1, 1, 1, 1, 0, 1, 1, 1, 1, 1, ]],
    # ]) * 4 + 1  # refractive index
    #
    # # case 4
    # grating_type = 2
    #
    # thickness, period = [1120.], [1000, 1000]
    # ucell = np.array(
    #     [
    #         [
    #             [3.5, 1.2, 1.5, 1.2, 3.3],
    #             [3.1, 1.5, 1.5, 1.4, 3.1],
    #         ],
    #     ]
    # )

    type_complex = 0
    device = 0
    return grating_type, pol, n_I, n_II, theta, phi, psi, wavelength, thickness, period, fourier_order, \
           type_complex, device, ucell


def optimize_jax():
    import jax
    import jax.numpy as jnp

    grating_type, pol, n_I, n_II, theta, phi, psi, wavelength, thickness, period, fourier_order, \
    type_complex, device, ucell = load_setting()

    mee = call_mee(backend=1, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi,
                   fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell,
                   thickness=thickness, device=device,
                   type_complex=type_complex, perturbation=1E-10)
    ucell = mee.ucell

    @jax.grad
    def grad_loss(ucell):
        mee.ucell = ucell
        de_ri, de_ti, _, _, _ = mee._conv_solve()
        try:
            loss = de_ti[de_ti.shape[0] // 2, de_ti.shape[1] // 2]
        except:
            loss = de_ti[de_ti.shape[0] // 2]
        return loss

    def grad_numerical(ucell, delta):
        grad_arr = jnp.zeros(ucell.shape, dtype=ucell.dtype)
        for layer in range(ucell.shape[0]):
            for r in range(ucell.shape[1]):
                for c in range(ucell.shape[2]):
                    ucell_delta_m = ucell.at[layer, r, c].set(ucell[layer, r, c] - delta)
                    mee.ucell = ucell_delta_m
                    de_ri_delta_m, de_ti_delta_m, _, _, _ = mee._conv_solve()
                    ucell_delta_p = ucell.at[layer, r, c].set(ucell[layer, r, c] + delta)
                    mee.ucell = ucell_delta_p
                    de_ri_delta_p, de_ti_delta_p, _, _, _ = mee._conv_solve()
                    try:
                        grad_numeric = \
                            (de_ti_delta_p[de_ti_delta_p.shape[0] // 2, de_ti_delta_p.shape[1] // 2]
                             - de_ti_delta_m[de_ti_delta_p.shape[0] // 2, de_ti_delta_p.shape[1] // 2]) / (2 * delta)
                    except:
                        grad_numeric = \
                            (de_ti_delta_p[de_ti_delta_p.shape[0] // 2]
                             - de_ti_delta_m[de_ti_delta_p.shape[0] // 2]) / (2 * delta)
                    grad_arr = grad_arr.at[layer, r, c].set(grad_numeric)

        return grad_arr

    grad_ad = grad_loss(ucell)
    print('JAX grad_ad:\n', grad_ad)
    grad_nume = grad_numerical(ucell, 1E-6)
    print('JAX grad_numeric:\n', grad_nume)
    print('JAX norm: ', jnp.linalg.norm(grad_nume - grad_ad) / grad_nume.size)


def optimize_torch():
    """
    out of date.
    Will be updated.
    """
    import torch

    grating_type, pol, n_I, n_II, theta, phi, psi, wavelength, thickness, period, fourier_order, \
    type_complex, device, ucell = load_setting()

    tmee = call_mee(backend=2, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi,
                    fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell,
                    thickness=thickness, device=device,
                    type_complex=type_complex, )
    tmee.ucell.requires_grad = True
    de_ri, de_ti = tmee.conv_solve()

    try:
        loss = de_ti[de_ti.shape[0] // 2, de_ti.shape[1] // 2]
    except:
        loss = de_ti[de_ti.shape[0] // 2]

    loss.backward()
    grad_ad = tmee.ucell.grad

    def grad_numerical(ucell, delta):
        ucell.requires_grad = False
        grad_arr = torch.zeros(ucell.shape, dtype=ucell.dtype)

        for layer in range(ucell.shape[0]):
            for r in range(ucell.shape[1]):
                for c in range(ucell.shape[2]):
                    ucell_delta_m = deepcopy(ucell)
                    ucell_delta_m[layer, r, c] -= delta
                    tmee.ucell = ucell_delta_m
                    de_ri_delta_m, de_ti_delta_m = tmee.conv_solve()

                    ucell_delta_p = deepcopy(ucell)
                    ucell_delta_p[layer, r, c] += delta
                    tmee.ucell = ucell_delta_p
                    de_ri_delta_p, de_ti_delta_p = tmee.conv_solve()
                    try:
                        grad_numeric = \
                            (de_ti_delta_p[de_ti_delta_p.shape[0] // 2, de_ti_delta_p.shape[1] // 2]
                             - de_ti_delta_m[de_ti_delta_p.shape[0] // 2, de_ti_delta_p.shape[1] // 2]) / (2 * delta)
                    except:
                        grad_numeric = \
                            (de_ti_delta_p[de_ti_delta_p.shape[0] // 2]
                             - de_ti_delta_m[de_ti_delta_p.shape[0] // 2]) / (2 * delta)
                    grad_arr[layer, r, c] = grad_numeric

        return grad_arr

    grad_nume = grad_numerical(tmee.ucell, 1E-6)
    print('Torch grad_ad:\n', grad_ad)
    print('Torch grad_numeric:\n', grad_nume)
    print('torch.norm: ', torch.linalg.norm(grad_nume - grad_ad) / grad_nume.numel())


if __name__ == '__main__':
    try:
        print('JaxMeent')
        optimize_jax()
    except Exception as e:
        print('JaxMeent has problem. Do you have JAX?')
        print(e)

    try:
        print('TorchMeent')
        optimize_torch()
    except Exception as e:
        print('TorchMeent has problem. Do you have PyTorch?')
        print(e)
