import numpy as np

from solver.LalanneClass import LalanneBase


class JLABCode(LalanneBase):
    def __init__(self, grating_type, n_I=1, n_II=1, theta=0, phi=0, psi=0, fourier_order=10, period=0.7,
                 wls=np.linspace(0.5, 2.3, 400), polarization=0, patterns=None, thickness=None):

        super().__init__(grating_type, n_I, n_II, theta, phi, psi, fourier_order, period, wls, polarization, patterns,
                         thickness)

    def run_1d(self):
        refl, tran = self.lalanne_1d()

        center = refl.shape[1]//2
        tran_cut = tran[:, center-1:center+2]

        return tran_cut


if __name__ == '__main__':
    JLABCode(0)
