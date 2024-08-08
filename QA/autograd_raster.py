import warnings
import jax
import jax.numpy as jnp
import torch

import numpy as np

from copy import deepcopy

from meent import call_mee


def load_setting():
    pol = 1  # 0: TE, 1: TM

    n_top = 1  # n_incidence
    n_bot = 1  # n_transmission

    theta = 0 * np.pi / 180
    phi = 0 * np.pi / 180
    psi = 0 * np.pi / 180 if pol else 90 * np.pi / 180

    wavelength = 900

    fto = [2, 2]

    # case 1
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

    # Case 4
    thickness = [1120]

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
    ucell = ucell.real

    type_complex = 0
    device = 0
    return pol, n_top, n_bot, theta, phi, psi, wavelength, thickness, period, fto, type_complex, device, ucell


def optimize_jax(setting):
    pol, n_top, n_bot, theta, phi, psi, wavelength, thickness, period, fto, \
        type_complex, device, ucell = setting

    mee = call_mee(backend=1, pol=pol, n_top=n_top, n_bot=n_bot, theta=theta, phi=phi,
                   fto=fto, wavelength=wavelength, period=period, ucell=ucell,
                   thickness=thickness, device=device,
                   type_complex=type_complex)
    ucell = mee.ucell

    @jax.grad
    def grad_loss(ucell):
        mee.ucell = ucell
        # de_ri, de_ti, _, _, _ = mee._conv_solve()
        de_ri, de_ti = mee.conv_solve()
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
                    # de_ri_delta_m, de_ti_delta_m, _, _, _ = mee._conv_solve()
                    de_ri_delta_m, de_ti_delta_m = mee.conv_solve()
                    ucell_delta_p = ucell.at[layer, r, c].set(ucell[layer, r, c] + delta)
                    mee.ucell = ucell_delta_p
                    # de_ri_delta_p, de_ti_delta_p, _, _, _ = mee._conv_solve()
                    de_ri_delta_p, de_ti_delta_p = mee.conv_solve()
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


def optimize_torch(setting):
    """
    out of date.
    Will be updated.
    """

    pol, n_top, n_bot, theta, phi, psi, wavelength, thickness, period, fto, \
        type_complex, device, ucell = setting

    tmee = call_mee(backend=2, pol=pol, n_top=n_top, n_bot=n_bot, theta=theta, phi=phi,
                    fto=fto, wavelength=wavelength, period=period, ucell=ucell,
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
    setting = load_setting()

    print('JaxMeent')
    optimize_jax(setting)

    print('TorchMeent')
    optimize_torch(setting)
