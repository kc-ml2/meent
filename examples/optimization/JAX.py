import numpy
import jax.numpy as np
import time

from jax import grad

from meent.rcwa import RCWA


class RCWAOptimizer:

    def __init__(self, gt, model):
        self.gt = gt
        self.model = model
        pass

    def get_difference(self):
        spectrum_gt = np.hstack(self.gt.spectrum_R, self.gt.spectrum_T)
        spectrum_model = np.hstack(self.model.spectrum_R, self.model.spectrum_T)
        residue = spectrum_model - spectrum_gt
        loss = np.linalg.norm(residue)


if __name__ == '__main__':

    def loss(thick):
        grating_type = 0
        pol = 0

        n_I = 1
        n_II = 1

        theta = 20
        phi = 20
        psi = 0 if pol else 90

        wls = np.linspace(500, 2300, 1)
        fourier_order = 10

        # Ground Truth
        period = np.array([700])
        thickness = np.array([1120])
        cell = np.array([[[3.48 ** 2, 3.48 ** 2, 3.48 ** 2, 1, 1, 1, 1, 1, 1, 1]]])
        ground_truth = RCWA(grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
                            fourier_order=fourier_order, wls=wls, period=period, patterns=cell, thickness=thickness)

        # Test
        thickness = np.array([thick])

        test = RCWA(grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
                    fourier_order=fourier_order, wls=wls, period=period, patterns=cell, thickness=thickness)

        a, b = ground_truth.jax_test()
        # ground_truth.plot()

        c, d = test.jax_test()
        # test.plot()
        gap = np.linalg.norm(test.spectrum_r - ground_truth.spectrum_r)
        print('gap:', gap.primal)
        return gap



    grad_loss = grad(loss)
    print('grad:', grad_loss(300.))
    print('grad:', grad_loss(600.))
    print('grad:', grad_loss(1110.))
    print('grad:', grad_loss(1120.))
    print('grad:', grad_loss(1130.))


    print('end')
