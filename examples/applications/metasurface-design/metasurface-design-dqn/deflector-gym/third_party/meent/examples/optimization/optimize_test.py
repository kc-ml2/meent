import numpy as np
import jax
import jax.numpy as jnp
import time

from jax import grad, vmap

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

    aa = jnp.array(1100, dtype='float32')
    cc = jnp.array(1E-4, dtype='float32')  # OK

    print(aa-cc)

    aa = jnp.array(1100, dtype='float32')
    cc = jnp.array(1E-5, dtype='float32')  # FAIL

    print(aa-cc)

    aa = np.array(1100, dtype='float32')
    cc = np.array(1E-5, dtype='float32')  # FAIL

    print(aa-cc)

    aa = np.array(1100, dtype='float64')
    cc = np.array(1E-5, dtype='float64')  # FAIL

    print(aa-cc)


    def loss(thick):
        grating_type = 0
        pol = 0

        n_I = 1
        n_II = 1

        theta = 20
        phi = 20
        psi = 0 if pol else 90

        wls = jnp.linspace(500, 2300, 1)
        fourier_order = 10

        # Ground Truth
        period = jnp.array([700])
        thickness = jnp.array([1120])
        cell = jnp.array([[[3.48 ** 2, 3.48 ** 2, 3.48 ** 2, 1, 1, 1, 1, 1, 1, 1]]])
        ground_truth = call_solver(mode=1, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
                            fourier_order=fourier_order, wls=wls, period=period, patterns=cell, thickness=thickness)

        # Test
        thickness = jnp.array([thick])

        test = call_solver(mode=1, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
                    fourier_order=fourier_order, wls=wls, period=period, patterns=cell, thickness=thickness)

        a, b = ground_truth.jax_test()

        c, d = test.jax_test()
        gap = jnp.linalg.norm(test.spectrum_r - ground_truth.spectrum_r)
        # print('gap:', gap.primal)
        return gap

    grad_loss = grad(loss)
    print('grad:', grad_loss(300.))
    print('grad:', grad_loss(600.))
    print('grad:', grad_loss(1110.))
    print('grad:', grad_loss(1120.))
    print('grad:', grad_loss(1130.))

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
        # print(x)
        # print(grad_loss(x))
        return x - 0.05*grad_loss(x)*x

    domain = jnp.linspace(1100, 1200, num=1)

    vfungd = vmap(mingd)
    # Recurrent loop of gradient descent
    for epoch in range(1000):
        domain = vfungd(domain)
        # print(domain)

    minfunc = vmap(loss)
    minimums = minfunc(domain)

    arglist = nanargmin(minimums)
    argmin = domain[arglist]
    minimum = minimums[arglist]

    print("The minimum is {} the argmin is {}".format(minimum, argmin))

    print('end')
