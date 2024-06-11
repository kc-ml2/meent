import time
import matplotlib.pyplot as plt

import numpy as np
from JLAB.solver import JLABCode
from benchmark.interface.Reticolo import Reticolo


class RetiMeent:
    def __init__(self, grating_type, n_air, n_si, n_glass, theta, phi, pol, thickness, deflected_angle, pattern, ucell, wls, fourier_order,
                 period):

        self.grating_type = grating_type
        self.n_air = n_air
        self.n_si = n_si
        self.n_glass = n_glass
        self.theta = theta
        self.phi = phi
        self.pol = pol
        self.thickness = thickness
        self.deflected_angle = deflected_angle
        self.pattern = pattern
        self.ucell = ucell
        self.wls = wls
        self.fourier_order = fourier_order
        self.period = period

        self.fourier_array = []
        self.reti_r = []
        self.reti_t = []
        self.meent_r = []
        self.meent_t = []

    def acs_run_meent(self):
        # patterns = [[self.n_si, self.n_air, self.pattern]]

        meent = JLABCode(grating_type=self.grating_type,
                         n_I=self.n_glass, n_II=self.n_air, theta=self.theta, phi=self.phi,
                         fourier_order=self.fourier_order, period=self.period,
                         wls=self.wls, pol=self.pol,
                         patterns=self.pattern, ucell=self.ucell, thickness=self.thickness)

        # poi, refl, tran = meent.reproduce_acs(patterns)
        poi, refl, tran = meent.reproduce_acs_cell(self.n_si, self.n_air)

        return poi, refl, tran

    def acs_run_reti(self):

        textures = profile = None

        reti = Reticolo(grating_type=self.grating_type,
                        n_I=self.n_air, n_II=self.n_glass, theta=self.theta, phi=self.phi, fourier_order=self.fourier_order, period=self.period,
                        wls=self.wls, pol=self.pol,
                        textures=textures, profile=profile, thickness=self.thickness, deflected_angle=self.deflected_angle,
                        engine_type='octave')

        poi, refl, tran = reti.run_acs(self.pattern, self.n_si)

        return poi, refl, tran

    def fourier_order_sweep(self, fourier_array):

        self.reti_r, self.reti_t, self.meent_r, self.meent_t = [], [], [], []
        self.fourier_array = fourier_array

        fourier_order = self.fourier_order

        for f_order in self.fourier_array:
            self.fourier_order = f_order
            a = self.acs_run_reti()
            b = self.acs_run_meent()

            self.reti_r.append(a[1])
            self.reti_t.append(a[2])
            self.meent_r.append(b[1])
            self.meent_t.append(b[2])

        self.fourier_order = fourier_order

        self.reti_r = np.array(self.reti_r)
        self.reti_t = np.array(self.reti_t)
        self.meent_r = np.array(self.meent_r)
        self.meent_t = np.array(self.meent_t)

        return self.reti_r, self.reti_t, self.meent_r, self.meent_t

    def fourier_order_sweep_plot(self):
        cut = 40

        figs, axes = plt.subplots(1, 3)

        axes[0].axvline(cut, c='r')
        axes[1].axvline(cut, c='r')
        axes[2].axvline(cut, c='r')

        axes[0].plot(self.fourier_array, self.reti_r[:, 0], marker='X')
        axes[0].plot(self.fourier_array, self.meent_r[:, 0], marker='.', linestyle='dashed')
        axes[1].plot(self.fourier_array, self.reti_r[:, 1], marker='X')
        axes[1].plot(self.fourier_array, self.meent_r[:, 1], marker='.', linestyle='dashed')
        axes[2].plot(self.fourier_array, self.reti_r[:, 2], marker='X')
        axes[2].plot(self.fourier_array, self.meent_r[:, 2], marker='.', linestyle='dashed')
        plt.show()

        figs, axes = plt.subplots(1, 3)

        axes[0].axvline(cut, c='r')
        axes[1].axvline(cut, c='r')
        axes[2].axvline(cut, c='r')

        axes[0].plot(self.fourier_array, self.reti_t[:, 0], marker='X')
        axes[0].plot(self.fourier_array, self.meent_t[:, 0], marker='.', linestyle='dashed')
        axes[1].plot(self.fourier_array, self.reti_t[:, 1], marker='X')
        axes[1].plot(self.fourier_array, self.meent_t[:, 1], marker='.', linestyle='dashed')
        axes[2].plot(self.fourier_array, self.reti_t[:, 2], marker='X')
        axes[2].plot(self.fourier_array, self.meent_t[:, 2], marker='.', linestyle='dashed')
        plt.show()

    def fourier_order_sweep_meent_2d(self, fourier_array):

        meent_r, meent_t = [], []

        fourier_order = self.fourier_order

        for f_order in fourier_array:
            self.fourier_order = f_order
            b = self.acs_run_meent()

            meent_r.append(b[1])
            meent_t.append(b[2])

        meent_r = np.array(meent_r)
        meent_t = np.array(meent_t)

        n_row, n_col = meent_t.shape[1], meent_t.shape[2]

        for i in range(n_row):
            for j in range(n_col):
                plt.plot(fourier_array, meent_r[:, i, j], marker='x')
                plt.plot(fourier_array, meent_t[:, i, j], marker='x')
                plt.title(f'Diffraction efficiency, {i-1}, {j-1} order')
                plt.legend(['reflectance', ' transmittance'])
                plt.show()

        self.fourier_order = fourier_order


if __name__ == '__main__':
    n_air = 1
    n_si = 3.5
    n_glass = 1.45
    theta = 0
    phi = 0
    pol = 1
    thickness = [325]
    wls = np.linspace(900, 900, 1)
    deflected_angle = 60

    # 1D
    grating_type = 0

    fourier_order = 40
    period = abs(wls / np.sin(deflected_angle / 180 * np.pi))
    pattern = np.array([1., 1., 1., -1., -1., -1., -1., -1., -1., -1.])
    ucell = np.array([[pattern]])

    AA = RetiMeent(grating_type, n_air, n_si, n_glass, theta, phi, pol, thickness, deflected_angle, pattern, ucell, wls, fourier_order,
                   period)

    reti_abseff, reti_ref, reti_tra = AA.acs_run_reti()
    meent_abseff, meent_ref, meent_tra = AA.acs_run_meent()

    print('reticolo result:, ', reti_abseff, reti_ref, reti_tra)
    print('meent result: ', meent_abseff, meent_ref, meent_tra)

    # Fourier order sweep

    fourier_array = [1, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180]
    AA.fourier_order_sweep(fourier_array)
    AA.fourier_order_sweep_plot()

    # Time comparison

    t_reti = 0
    for i in range(100):
        t0 = time.time()
        AA.acs_run_reti()
        t_reti += time.time() - t0

    t_meent = 0
    for i in range(100):
        t0 = time.time()
        AA.acs_run_meent()
        t_meent += time.time() - t0

    print(t_reti/100, t_meent/100)

    # 2D
    grating_type = 2
    fourier_order = 2
    pattern = np.array([
        [1., 1., 1., -1., -1., -1., -1., -1., -1., -1.],
        [1., 1., 1., -1., -1., -1., -1., -1., -1., -1.],
        [1., 1., 1., -1., -1., -1., -1., -1., -1., -1.],
        [1., 1., 1., -1., -1., -1., -1., -1., -1., -1.],
        [1., 1., 1., -1., -1., -1., -1., -1., -1., -1.],
        [1., 1., 1., -1., -1., -1., -1., -1., -1., -1.],
        [1., 1., 1., -1., -1., -1., -1., -1., -1., -1.],
        [1., 1., 1., -1., -1., -1., -1., -1., -1., -1.],
        [1., 1., 1., -1., -1., -1., -1., -1., -1., -1.],
        [1., 1., 1., -1., -1., -1., -1., -1., -1., -1.]
        ])
    ucell = np.array([pattern])

    period = [abs(wls / np.sin(deflected_angle / 180 * np.pi)),
              abs(wls / np.sin(deflected_angle / 180 * np.pi))]

    AA = RetiMeent(grating_type,n_air, n_si, n_glass, theta, phi, pol, thickness, deflected_angle, pattern, ucell, wls, fourier_order,
                   period)

    abseff, de_ri, de_ti = AA.acs_run_meent()

    # print('meent result: ', res_meent)
    for i in range(len(de_ti)):
        print(de_ti[i].round(4))

    # Fourier order sweep

    fourier_array = [1, 2, 3, 4, 5, 6]
    AA.fourier_order_sweep_meent_2d(fourier_array)
