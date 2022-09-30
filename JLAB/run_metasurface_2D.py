import time
import matplotlib.pyplot as plt

import numpy as np
from JLAB.JLAB import JLABCode
from benchmark.interface.Reticolo import Reticolo


class RetiMeent:
    def __init__(self, n_air, n_si, n_glass, theta, phi, pol, thickness, deflected_angle, pattern, wls, fourier_order,
                 period):
        self.n_air = n_air
        self.n_si = n_si
        self.n_glass = n_glass
        self.theta = theta
        self.phi = phi
        self.pol = pol
        self.thickness = thickness
        self.deflected_angle = deflected_angle
        self.pattern = pattern
        self.wls = wls
        self.fourier_order = fourier_order
        self.period = period

    def acs_run_meent(self):
        patterns = [[self.n_si, self.n_air, self.pattern]]

        meent = JLABCode(grating_type=0,
                         n_I=self.n_glass, n_II=self.n_air, theta=self.theta, phi=self.phi,
                         fourier_order=self.fourier_order, period=self.period,
                         wls=self.wls, pol=self.pol,
                         patterns=patterns, thickness=self.thickness)

        poi, refl, tran = meent.reproduce_acs(patterns)

        return poi, refl, tran

    def acs_run_reti(self):

        textures = profile = None

        reti = Reticolo(grating_type=0,
                        n_I=self.n_air, n_II=self.n_glass, theta=self.theta, phi=self.phi, fourier_order=self.fourier_order, period=self.period,
                        wls=self.wls, pol=self.pol,
                        textures=textures, profile=profile, thickness=self.thickness, deflected_angle=self.deflected_angle,
                        engine_type='octave')

        poi, refl, tran = reti.run_acs(self.pattern, self.n_si)

        return poi, refl, tran

    def make_spectrum(self):

        textures = profile = None

        reti = Reticolo(grating_type=0,
                        n_I=self.n_air, n_II=self.n_glass, theta=self.theta, phi=self.phi, fourier_order=self.fourier_order, period=self.period,
                        wls=self.wls, pol=self.pol,
                        textures=textures, profile=profile, thickness=self.thickness, deflected_angle=self.deflected_angle,
                        engine_type='octave')

        reti.run_acs_loop_wavelength(self.pattern, self.deflected_angle, n_si=self.n_si)

        patterns = [[self.n_si, self.n_air, self.pattern]]

        meent = JLABCode(grating_type=0,
                         n_I=self.n_glass, n_II=self.n_air, theta=self.theta, phi=self.phi,
                         fourier_order=self.fourier_order, period=self.period,
                         wls=self.wls, pol=self.pol,
                         patterns=patterns, thickness=self.thickness)

        meent.reproduce_acs_loop_wavelength(patterns, self.deflected_angle)

    def fourier_order_sweep(self, fourier_array):

        reti_r, reti_t, meent_r, meent_t = [], [], [], []

        fourier_order = self.fourier_order

        for f_order in fourier_array:
            self.fourier_order = f_order
            a = self.acs_run_reti()
            b = self.acs_run_meent()

            reti_r.append(a[1])
            reti_t.append(a[2])
            meent_r.append(b[1])
            meent_t.append(b[2])

        self.fourier_order = fourier_order

        reti_r = np.array(reti_r)
        reti_t = np.array(reti_t)
        meent_r = np.array(meent_r)
        meent_t = np.array(meent_t)

        for i in range(3):
            plt.plot(fourier_array, reti_r[:, i], marker='x')
            plt.plot(fourier_array, meent_r[:, i], marker='x')
            plt.title(f'reflectance, {i-1}order')
            plt.legend(['reti', ' meent'])
            plt.show()

        for i in range(3):
            plt.plot(fourier_array, reti_t[:, i], marker='x')
            plt.plot(fourier_array, meent_t[:, i], marker='x')
            plt.title(f'transmittance, {i-1}order')
            plt.legend(['reti', ' meent'])
            plt.show()

        plt.hist(np.array([meent_r-reti_r, meent_t-reti_t]).flatten())
        plt.title('histogram of errors')
        plt.show()


if __name__ == '__main__':

    n_air = 1
    n_si = 3.5
    n_glass = 1.45
    theta = 0
    phi = 0

    pol = 1

    thickness = [325]

    deflected_angle = 60

    # pattern = np.load('structure.npy')
    pattern = np.array([[1., 1., 1., -1.],
                        [1., 1., 1., 1.],
                        [1., 1., 1., 1.],
                        [1., 1., 1., -1.]])

    wls = np.linspace(900, 900, 1)
    fourier_order = 2
    period = abs(wls / np.sin(deflected_angle / 180 * np.pi))

    AA = RetiMeent(n_air, n_si, n_glass, theta, phi, pol, thickness, deflected_angle, pattern, wls, fourier_order,
                   period)

    res_reti = AA.acs_run_reti()
    res_meent = AA.acs_run_meent()

    print('reticolo result:, ', res_reti)
    print('meent result: ', res_meent)

    # # Fourier order sweep

    # fourier_array = [1, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180]
    # fourier_array = [1, 10, 20, 40, 60, 80]
    # AA.fourier_order_sweep(fourier_array)

    # # Time comparison

    # t_reti = 0
    # for i in range(100):
    #     t0 = time.time()
    #     AA.acs_run_reti()
    #     t_reti += time.time() - t0
    #
    # t_meent = 0
    # for i in range(100):
    #     t0 = time.time()
    #     AA.acs_run_meent()
    #     t_meent += time.time() - t0
    #
    # print(t_reti/100, t_meent/100)

