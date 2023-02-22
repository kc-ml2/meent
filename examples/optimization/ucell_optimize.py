try:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
except:
    pass

import numpy as np
import jax
import jax.numpy as jnp
import time

from meent.rcwa import call_solver
import torch


class RCWAOptimizer:

    def __init__(self, gt, model):
        self.gt = gt
        self.model = model
        pass

    def get_difference(self):
        spectrum_gt = jnp.hstack(self.gt.spectrum_R, self.gt.spectrum_T)
        spectrum_model = jnp.hstack(self.model.spectrum_R, self.model.spectrum_T)
        residue = spectrum_model - spectrum_gt
        loss = jnp.linalg.norm(residue)


def load_setting(mode_key, dtype, device):
    grating_type = 2

    pol = 1  # 0: TE, 1: TM

    n_I = 1  # n_incidence
    n_II = 1  # n_transmission

    theta = 0 * np.pi / 180
    phi = 0 * np.pi / 180
    psi = 0 * np.pi / 180 if pol else 90 * np.pi / 180

    wavelength = 900

    ucell_materials = [1, 3.48]
    fourier_order = 2

    thickness, period = [1120.], [1000, 1000]

    ucell = np.array(
        [[
            [3., 1., 1., 1., 3.],
            [3., 1., 1., 1., 3.],
            [3., 1., 1., 1., 3.],
            [3., 1., 1., 1., 3.],
            [3., 1., 1., 1., 3.],
        ]]
    )

    if mode_key == 0:
        device = 0
        type_complex = np.complex128 if dtype == 0 else np.complex64
        ucell = ucell.astype(type_complex)

    elif mode_key == 1:  # JAX
        jax.config.update('jax_platform_name', 'cpu') if device == 0 else jax.config.update('jax_platform_name', 'gpu')

        if dtype == 0:
            jax.config.update("jax_enable_x64", True)
            type_complex = jnp.complex128
            ucell = ucell.astype(jnp.float64)
            ucell = jnp.array(ucell, dtype=jnp.float64)

        else:
            type_complex = jnp.complex64
            ucell = ucell.astype(jnp.float32)
            ucell = jnp.array(ucell, dtype=jnp.float32)


    else:  # Torch
        device = torch.device('cpu') if device == 0 else torch.device('cuda')
        type_complex = torch.complex128 if dtype == 0 else torch.complex64

        if dtype == 0:
            ucell = torch.tensor(ucell, dtype=torch.float64, device=device)
        else:
            ucell = torch.tensor(ucell, dtype=torch.float32, device=device)

    return grating_type, pol, n_I, n_II, theta, phi, psi, wavelength, thickness, ucell_materials, period, fourier_order,\
           type_complex, device, ucell


def optimize_jax_thickness(mode_key, dtype, device):
    from meent.on_jax.convolution_matrix import to_conv_mat

    grating_type, pol, n_I, n_II, theta, phi, psi, wavelength, thickness, ucell_materials, period, fourier_order, \
    type_complex, device, ucell = load_setting(mode_key, dtype, device)

    solver = call_solver(mode_key, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi,
                         psi=psi, fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell,
                         ucell_materials=ucell_materials, thickness=thickness, device=device,
                         type_complex=type_complex, )

    E_conv_all = to_conv_mat(ucell, fourier_order, type_complex=type_complex)
    o_E_conv_all = to_conv_mat(1 / ucell, fourier_order, type_complex=type_complex)
    de_ri_gt, de_ti_gt, _, _, _ = solver.solve(wavelength, E_conv_all, o_E_conv_all)

    thickness = jnp.array([1000.])

    @jax.grad
    def grad_loss(thickness):

        # E_conv_all = to_conv_mat(ucell, fourier_order, type_complex=type_complex)
        # o_E_conv_all = to_conv_mat(1 / ucell, fourier_order, type_complex=type_complex)
        solver.thickness = thickness
        de_ri, de_ti, _, _, _ = solver.solve(wavelength, E_conv_all, o_E_conv_all)

        loss = jnp.linalg.norm(de_ti_gt - de_ti) + jnp.linalg.norm(de_ri_gt - de_ri)
        print(thickness.primal, loss.primal)
        return loss

    print('grad:', grad_loss(thickness))

    def mingd(x):
        lr = 0.1
        gd = grad_loss(x)

        res = x - lr*gd*x
        return res

    # Recurrent loop of gradient descent
    for i in range(1000):
        # ucell = vfungd(ucell)
        thickness = mingd(thickness)

    print(thickness)
    # E_conv_all = to_conv_mat(ucell, fourier_order, type_complex=type_complex)
    # o_E_conv_all = to_conv_mat(1 / ucell, fourier_order, type_complex=type_complex)
    solver.thickness = thickness
    de_ri, de_ti, _, _, _ = solver.solve(wavelength, E_conv_all, o_E_conv_all)
    print(de_ti)


def optimize_jax_ucell(mode_key, dtype, device):
    from meent.on_jax.convolution_matrix import to_conv_mat

    grating_type, pol, n_I, n_II, theta, phi, psi, wavelength, thickness, ucell_materials, period, fourier_order, \
    type_complex, device, ucell = load_setting(mode_key, dtype, device)


    solver = call_solver(mode_key, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi,
                         psi=psi, fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell,
                         ucell_materials=ucell_materials, thickness=thickness, device=device,
                         type_complex=type_complex, )
    ucell_gt = jnp.array(
        [[
            [3., 1.5, 1., 1.2, 3.],
            [3., 1.5, 1., 1.3, 3.],
            # [3., 1.5, 1., 1.4, 3.],
            # [3., 1.5, 1., 1.2, 3.],
            # [3., 1.5, 1., 1.9, 3.],
        ]], dtype=jnp.float64
    )
    E_conv_all = to_conv_mat(ucell_gt, fourier_order, type_complex=type_complex)
    o_E_conv_all = to_conv_mat(1 / ucell_gt, fourier_order, type_complex=type_complex)
    de_ri_gt, de_ti_gt, _, _, _ = solver.solve(wavelength, E_conv_all, o_E_conv_all)

    # val = np.array([3.1])
    ucell = jnp.array(
        [[
            [3.1, 1.5, 1., 1.2, 3.],
            [3., 1.5, 1., 1.3, 3.],
            # [3., 1.5, 1., 1.4, 3.],
            # [3., 1.5, 1., 1.2, 3.],
            # [3., 1.5, 1., 1.9, 3.],
        ]], dtype=jnp.float64
    )
    @jax.grad
    def grad_loss(ucell):

        # ucell = jnp.array(
        #     [[
        #         [3., 1.5, 1., 1.2, 3.],
        #         [3., 1.5, 1., 1.3, 3.],
        #         # [3., 1.5, 1., 1.4, 3.],
        #         # [3., 1.5, 1., 1.2, 3.],
        #         # [3., 1.5, 1., 1.9, 3.],
        #     ]], dtype=jnp.float64
        # )
        #
        # ucell = ucell.at[0,0,0].set(val[0])

        E_conv_all = to_conv_mat(ucell, fourier_order, type_complex=type_complex)
        o_E_conv_all = to_conv_mat(1 / ucell, fourier_order, type_complex=type_complex)
        de_ri, de_ti, _, _, _ = solver.solve(wavelength, E_conv_all, o_E_conv_all)

        loss = jnp.linalg.norm(de_ti_gt - de_ti) + jnp.linalg.norm(de_ri_gt - de_ri)
        print(loss.primal)
        return loss

    print('grad:', grad_loss(ucell))

    def mingd(x):
        lr = 0.01
        gd = grad_loss(x)

        res = x - lr*gd*x
        # print(res)
        return res

    # Recurrent loop of gradient descent
    for i in range(100):
        # ucell = vfungd(ucell)
        ucell = mingd(ucell)

    print(ucell)
    # ucell = ucell.at[0,0,0].set(val[0])
    # E_conv_all = to_conv_mat(ucell, fourier_order, type_complex=type_complex)
    # o_E_conv_all = to_conv_mat(1 / ucell, fourier_order, type_complex=type_complex)
    # de_ri, de_ti = solver.solve(wavelength, E_conv_all, o_E_conv_all)
    # print(de_ti)


def optimize_torch(mode_key, dtype, device):
    """
    out of date.
    Will be updated.
    """
    from meent.on_torch.convolution_matrix import to_conv_mat

    grating_type, pol, n_I, n_II, theta, phi, psi, wavelength, thickness, ucell_materials, period, fourier_order, \
    type_complex, device, ucell = load_setting(mode_key, dtype, device)

    ucell.requires_grad = True

    solver = call_solver(mode_key, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi,
                         psi=psi, fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell,
                         ucell_materials=ucell_materials, thickness=thickness, device=device,
                         type_complex=type_complex, )

    opt = torch.optim.SGD([ucell], lr=1E-2)
    for i in range(100):
        E_conv_all = to_conv_mat(ucell, fourier_order)
        o_E_conv_all = to_conv_mat(1 / ucell, fourier_order)
        de_ri, de_ti, _, _, _ = solver.solve(wavelength, E_conv_all, o_E_conv_all)

        loss = -de_ti[3, 2]
        loss.backward()
        print(ucell.grad)
        opt.step()
        opt.zero_grad()
        print(loss)


if __name__ == '__main__':

    optimize_jax_thickness(1, 0, 0)
    optimize_jax_ucell(1, 0, 0)
    optimize_torch(2, 0, 0)

