import meent.on_numpy.convolution_matrix

try:
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
except:
    pass


import numpy as np
import jax
import jax.numpy as jnp
import time

from jax import grad, vmap

from meent.on_jax.convolution_matrix import put_permittivity_in_ucell, to_conv_mat, to_conv_mat_piecewise_constant

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


if __name__ == '__main__':
    t0 = time.time()

    grating_type = 2

    # Ground Truth
    pol = 1  # 0: TE, 1: TM

    n_I = 1  # n_incidence
    n_II = 1  # n_transmission

    theta = 0
    phi = 0
    psi = 0 if pol else 90

    wavelength = 900

    thickness_gt = [1120]
    ucell_materials = [1, 3.48]
    period = [1000, 1000]
    fourier_order = 3

    mode_key = 1
    device = 0
    dtype = 0

    ucell_gt = np.array(
        [[
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
        ]]
    )

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
        device = None

        if dtype == 0:
            type_complex = np.complex128
        else:
            type_complex = np.complex64

    elif mode_key == 1:
        # JAX
        if device == 0:
            jax.config.update('jax_platform_name', 'cpu')
        else:
            jax.config.update('jax_platform_name', 'gpu')

        if dtype == 0:
            from jax.config import config

            config.update("jax_enable_x64", True)
            type_complex = jnp.complex128
        else:
            type_complex = jnp.complex64

    else:
        # Torch
        if device == 0:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')

        if dtype == 0:
            type_complex = torch.complex128
        else:
            type_complex = torch.complex64

    solver = call_solver(mode_key, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi,
                         psi=psi,
                         fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell,
                         ucell_materials=ucell_materials,
                         thickness=thickness_gt, device=device, type_complex=type_complex, )

    E_conv_all = to_conv_mat(ucell, fourier_order, type_complex=type_complex)
    o_E_conv_all = to_conv_mat(1 / ucell, fourier_order, type_complex=type_complex)
    de_ri, de_ti = solver.solve(wavelength, E_conv_all, o_E_conv_all)

    E_conv_all1 = to_conv_mat_piecewise_constant(ucell, fourier_order, type_complex=type_complex)
    o_E_conv_all1 = to_conv_mat_piecewise_constant(1 / ucell, fourier_order, type_complex=type_complex)
    de_ri1, de_ti1 = solver.solve(wavelength, E_conv_all1, o_E_conv_all1)

    solver1 = call_solver(0, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi,
                         psi=psi,
                         fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell,
                         ucell_materials=ucell_materials,
                         thickness=thickness_gt, device=device, type_complex=type_complex, )

    E_conv_all2 = meent.on_numpy.convolution_matrix.to_conv_mat(ucell, fourier_order)
    o_E_conv_all2 = meent.on_numpy.convolution_matrix.to_conv_mat(1/ucell, fourier_order)
    de_ri2, de_ti2 = solver1.solve(wavelength, E_conv_all2, o_E_conv_all2)


    E_conv_all3 = meent.on_numpy.convolution_matrix.to_conv_mat_piecewise_constant(ucell, fourier_order)
    o_E_conv_all3 = meent.on_numpy.convolution_matrix.to_conv_mat_piecewise_constant(1/ucell, fourier_order)
    de_ri3, de_ti3 = solver1.solve(wavelength, E_conv_all3, o_E_conv_all3)

    def loss(ucell):

        E_conv_all = to_conv_mat(ucell, fourier_order, type_complex=type_complex)
        o_E_conv_all = to_conv_mat(1 / ucell, fourier_order, type_complex=type_complex)
        de_ri, de_ti = solver.solve(wavelength, E_conv_all, o_E_conv_all)

        res = -de_ti[3,2]
        print(res)
        return res

    grad_loss = grad(loss)
    print('grad:', grad_loss(ucell))


    def mingd(x):

        lr = 0.01
        gd = grad_loss(x)

        res = x - lr*gd*x
        return res

    # Recurrent loop of gradient descent
    for i in range(1):
        # ucell = vfungd(ucell)
        ucell = mingd(ucell)

    print(ucell)



    ucell = torch.tensor(
        [[
            [3., 1., 1., 1., 3.],
            [3., 1., 1., 1., 3.],
            [3., 1., 1., 1., 3.],
            [3., 1., 1., 1., 3.],
            [3., 1., 1., 1., 3.],
        ]], dtype=torch.float64, requires_grad=True)

    type_complex = torch.complex128
    device = torch.device('cpu')

    solver2 = call_solver(2, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi,
                         psi=psi,
                         fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell,
                         ucell_materials=ucell_materials,
                         thickness=thickness_gt, device=device, type_complex=type_complex, )

    E_conv_all4 = meent.on_torch.convolution_matrix.to_conv_mat(ucell, fourier_order)
    o_E_conv_all4 = meent.on_torch.convolution_matrix.to_conv_mat(1/ucell, fourier_order)
    de_ri4, de_ti4 = solver2.solve(wavelength, E_conv_all4, o_E_conv_all4)

    E_conv_all5 = meent.on_torch.convolution_matrix.to_conv_mat_piecewise_constant(ucell, fourier_order)
    o_E_conv_all5 = meent.on_torch.convolution_matrix.to_conv_mat_piecewise_constant(1/ucell, fourier_order)
    de_ri5, de_ti5 = solver2.solve(wavelength, E_conv_all5, o_E_conv_all5)

    opt = torch.optim.SGD([ucell], lr=1E-2)
    for i in range(1):
        E_conv_all = meent.on_torch.convolution_matrix.to_conv_mat(ucell, fourier_order, type_complex=type_complex)
        o_E_conv_all = meent.on_torch.convolution_matrix.to_conv_mat(1 / ucell, fourier_order, type_complex=type_complex)
        de_ri, de_ti = solver2.solve(wavelength, E_conv_all, o_E_conv_all)
        # loss = (torch.linalg.norm(de_ti - de_ti_gt) + torch.linalg.norm(de_ri-de_ri_gt)).sum()
        loss = -de_ti[3, 2]
        loss.backward()
        print(ucell.grad)
        opt.step()
        opt.zero_grad()
        print(loss)

    print(ucell)

