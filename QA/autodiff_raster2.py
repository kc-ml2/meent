import jax
import torch

import jax.numpy as jnp
import numpy as np

from time import time

from meent import call_mee


def load_setting():
    pol = 1  # 0: TE, 1: TM

    n_top = 1  # n_incidence
    n_bot = 1  # n_transmission

    theta = 0 * np.pi / 180
    phi = 0 * np.pi / 180

    wavelength = 900

    fto = [5, 5]

    period = [1000, 1000]
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

    setting = {'pol': pol, 'n_top': n_top, 'n_bot': n_bot, 'theta': theta, 'phi': phi, 'fto': fto,
               'wavelength': wavelength, 'period': period, 'ucell': ucell, 'thickness': thickness, 'device': device,
               'type_complex': type_complex}

    return setting


def optimize_jax(setting):
    ucell = setting['ucell']

    mee = call_mee(backend=1, **setting)

    @jax.jit
    def grad_loss(ucell):
        mee.ucell = ucell
        res = mee.conv_solve().res
        de_ri, de_ti = res.de_ri, res.de_ti

        loss = de_ti[de_ti.shape[0] // 2, de_ti.shape[1] // 2]

        return loss

    def grad_numerical(ucell, delta):
        grad_arr = jnp.zeros(ucell.shape, dtype=ucell.dtype)

        @jax.jit
        def compute(ucell):
            mee.ucell = ucell
            result = mee.conv_solve()
            de_ti = result.res.de_ti
            loss = de_ti[de_ti.shape[0] // 2, de_ti.shape[1] // 2]

            return loss

        for layer in range(ucell.shape[0]):
            for r in range(ucell.shape[1]):
                for c in range(ucell.shape[2]):
                    ucell_delta_m = ucell.copy()
                    ucell_delta_m[layer, r, c] -= delta
                    mee.ucell = ucell_delta_m
                    de_ti_delta_m = compute(ucell_delta_m, )

                    ucell_delta_p = ucell.copy()
                    ucell_delta_p[layer, r, c] += delta
                    mee.ucell = ucell_delta_p
                    de_ti_delta_p = compute(ucell_delta_p, )

                    grad_numeric = (de_ti_delta_p - de_ti_delta_m) / (2 * delta)
                    grad_arr = grad_arr.at[layer, r, c].set(grad_numeric)

        return grad_arr

    jax.grad(grad_loss)(ucell)  # Dry run for jit compilation. This is to make time comparison fair.
    t0 = time()
    grad_ad = jax.grad(grad_loss)(ucell)
    t_ad = time() - t0
    print('JAX grad_ad:\n', grad_ad)
    t0 = time()
    grad_nume = grad_numerical(ucell, 1E-6)
    t_nume = time() - t0
    print('JAX grad_numeric:\n', grad_nume)
    print('JAX norm of difference: ', jnp.linalg.norm(grad_nume - grad_ad) / grad_nume.size)
    return t_ad, t_nume


def optimize_torch(setting):
    mee = call_mee(backend=2, **setting)

    mee.ucell.requires_grad = True

    t0 = time()
    res = mee.conv_solve().res
    de_ri, de_ti = res.de_ri, res.de_ti

    loss = de_ti[de_ti.shape[0] // 2, de_ti.shape[1] // 2]

    loss.backward()
    grad_ad = mee.ucell.grad
    t_ad = time() - t0

    def grad_numerical(ucell, delta):
        ucell.requires_grad = False
        grad_arr = torch.zeros(ucell.shape, dtype=ucell.dtype)

        for layer in range(ucell.shape[0]):
            for r in range(ucell.shape[1]):
                for c in range(ucell.shape[2]):
                    ucell_delta_m = ucell.clone().detach()
                    ucell_delta_m[layer, r, c] -= delta
                    mee.ucell = ucell_delta_m
                    res = mee.conv_solve().res
                    de_ri_delta_m, de_ti_delta_m = res.de_ri, res.de_ti

                    ucell_delta_p = ucell.clone().detach()
                    ucell_delta_p[layer, r, c] += delta
                    mee.ucell = ucell_delta_p
                    res = mee.conv_solve().res
                    de_ri_delta_p, de_ti_delta_p = res.de_ri, res.de_ti

                    cy, cx = np.array(de_ti_delta_p.shape) // 2
                    grad_numeric = (de_ti_delta_p[cy, cx] - de_ti_delta_m[cy, cx]) / (2 * delta)
                    grad_arr[layer, r, c] = grad_numeric

        return grad_arr

    t0 = time()
    grad_nume = grad_numerical(mee.ucell, 1E-6)
    t_nume = time() - t0

    print('Torch grad_ad:\n', grad_ad)
    print('Torch grad_numeric:\n', grad_nume)
    print('torch.norm: ', torch.linalg.norm(grad_nume - grad_ad) / grad_nume.numel())
    return t_ad, t_nume


if __name__ == '__main__':
    setting = load_setting()

    print('JaxMeent')
    j_t_ad, j_t_nume = optimize_jax(setting)
    print('TorchMeent')
    t_t_ad, t_t_nume = optimize_torch(setting)

    print(f'Time for Backprop, JAX, AD: {j_t_ad} s, Numerical: {j_t_nume} s')
    print(f'Time for Backprop, Torch, AD: {t_t_ad} s, Numerical: {t_t_nume} s')
