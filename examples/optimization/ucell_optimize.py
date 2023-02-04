# import numpy as np
import jax
import jax.numpy as jnp
import time

from jax import grad, vmap

from examples.ex_ucell import load_ucell
from meent.on_jax.convolution_matrix import put_permittivity_in_ucell, to_conv_mat
# from meent.rcwa import RCWA

from meent.rcwa import call_solver

# from jax.config import config; config.update("jax_enable_x64", True)


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
    period = [100, 100]
    fourier_order = 2
    mode_key = 1
    device = 0
    dtype = 0

    # ucell_gt = load_ucell(grating_type)
    #
    # ucell = ucell_gt.copy()
    # ucell[0, 0, :] = 1

    ucell_gt = jnp.array(
        [[
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
        ]]
    )

    ucell = jnp.array(
        [[
            [3., 1., 1., 1., 3.],
            [3., 1., 1., 1., 3.],
            [3., 1., 1., 1., 3.],
            [3., 1., 1., 1., 3.],
            [3., 1., 1., 1., 3.],
        ]]
    )

    # ucell = ucell_gt.copy()
    # ucell[0, 0, :] = 1

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

    # ucell = put_permittivity_in_ucell(ucell, ucell_materials, self.mat_table, self.wavelength,
    #                                       type_complex=self.type_complex)

    solver.ucell = ucell_gt
    a, b = solver.run_ucell()
    # ucell = put_permittivity_in_ucell(ucell, self.ucell_materials, self.mat_table, self.wavelength,
    #                                   type_complex=self.type_complex)
    E_conv_all = to_conv_mat(ucell, fourier_order, type_complex=type_complex)
    o_E_conv_all = to_conv_mat(1 / ucell, fourier_order, type_complex=type_complex)

    de_ri, de_ti = solver.solve(wavelength, E_conv_all, o_E_conv_all)

    solver.ucell = ucell_gt
    a, b = solver.run_ucell()

    def loss(ucell):

        # solver.thickness = thickness_gt
        # ucell = put_permittivity_in_ucell(ucell, self.ucell_materials, self.mat_table, self.wavelength,
        #                                   type_complex=self.type_complex)
        E_conv_all = to_conv_mat(ucell, fourier_order, type_complex=type_complex)
        o_E_conv_all = to_conv_mat(1 / ucell, fourier_order, type_complex=type_complex)

        de_ri, de_ti = solver.solve(wavelength, E_conv_all, o_E_conv_all)

        # solver.thickness = [thickness]
        # solver.ucell = ucell
        # c, d = solver.run_ucell()

        # gap = jnp.linalg.norm(a - c)
        # print('gap:', gap.primal)
        res = de_ti[2,2]
        return res

    grad_loss = grad(loss)
    print('grad:', grad_loss(ucell))
    # print('grad:', grad_loss(600.))
    # print('grad:', grad_loss(1110.))
    # print('grad:', grad_loss(1120.))
    # print('grad:', grad_loss(1130.))

    import jax.numpy as np
    from jax import grad, jit, vmap
    from jax import random
    from jax import jacfwd, jacrev
    from jax.numpy import linalg

    from numpy import nanargmin, nanargmax

    # key = random.PRNGKey(42)

    # value_fn = lambda theta, state: jnp.dot(theta, state)
    # theta = jnp.array([0.1, -0.1, 0.])
    # # An example transition.
    # s_tm1 = jnp.array([1., 2., -1.])
    # r_t = jnp.array(1.)
    # s_t = jnp.array([2., 1., 0.])

    def mingd(x):

        lr = 0.05
        gd = grad_loss(x)

        res = x - lr*gd*x
        return res

    domain = [ucell, ucell+1, ucell+2]

    # vfungd = vmap(mingd)

    # Recurrent loop of gradient descent
    for i in range(50):
        # ucell = vfungd(ucell)
        ucell = mingd(ucell)
        print(ucell)

    minfunc = vmap(loss)
    minimums = minfunc(domain)

    arglist = nanargmin(minimums)
    argmin = domain[arglist]
    minimum = minimums[arglist]

    print("The minimum is {} the argmin is {}".format(minimum, argmin))
    print(time.time() - t0)
    print('end')
