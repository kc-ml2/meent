import time
import jax.numpy as jnp

from ._base import _BaseRCWA
from .convolution_matrix import to_conv_mat, find_n_index, fill_factor_to_ucell


class RCWAOpt(_BaseRCWA):
    def __init__(self, mode=0, grating_type=0, n_I=1., n_II=1., theta=0, phi=0, psi=0, fourier_order=40, period=(100,),
                 wls=jnp.linspace(900, 900, 1), pol=0, patterns=None, thickness=None, algo='TMM'):

        super().__init__(grating_type, n_I, n_II, theta, phi, psi, fourier_order, period, wls, pol, patterns,
                         thickness, algo)
        self.mode = mode
        self.spectrum_r, self.spectrum_t = None, None
        self.init_spectrum_array()

    def solve(self, wl, e_conv_all, o_e_conv_all):

        # TODO: !handle uniform layer

        if self.grating_type == 0:
            de_ri, de_ti = self.solve_1d(wl, e_conv_all, o_e_conv_all)
        elif self.grating_type == 1:
            de_ri, de_ti = self.solve_1d_conical(wl, e_conv_all, o_e_conv_all)
        elif self.grating_type == 2:
            de_ri, de_ti = self.solve_2d(wl, e_conv_all, o_e_conv_all)
        else:
            raise ValueError

        return de_ri.real, de_ti.real

    def loop_wavelength_fill_factor(self, wavelength_array=None):

        if wavelength_array is not None:
            self.wls = wavelength_array
            self.init_spectrum_array()

        for i, wl in enumerate(self.wls):

            ucell = fill_factor_to_ucell(self.patterns, wl, self.grating_type)
            e_conv_all = to_conv_mat(ucell, self.fourier_order)
            o_e_conv_all = to_conv_mat(1 / ucell, self.fourier_order)

            de_ri, de_ti = self.solve(wl, e_conv_all, o_e_conv_all)

            self.spectrum_r = self.spectrum_r.at[i].set(de_ri)
            self.spectrum_t = self.spectrum_t.at[i].set(de_ti)

        return self.spectrum_r, self.spectrum_t

    def loop_wavelength_ucell(self):
        # si = [[z_begin, z_end], [y_begin, y_end], [x_begin, x_end]]
        if self.grating_type == 0:
            cell = jnp.ones((2, 1, 10))
            si = [3.48, 0, 1, 0, 1, 0, 3]
            ox = [3.48, 1, 2, 0, 1, 0, 3]
        elif self.grating_type == 1:
            cell = jnp.ones((2, 1, 10))
            si = [3.48, 0, 1, 0, 1, 0, 3]
            ox = [3.48, 1, 2, 0, 1, 0, 3]
        elif self.grating_type == 2:
            cell = jnp.ones((2, 10, 10))
            si = [3.48, 0, 1, 0, 10, 0, 3]
            ox = [3.48, 1, 2, 0, 10, 0, 3]
        else:
            raise ValueError

        for i, wl in enumerate(self.wls):
            for material, z_begin, z_end, y_begin, y_end, x_begin, x_end in [si, ox]:
                n_index = find_n_index(material, wl) if type(material) == str else material
                cell = cell.at[z_begin:z_end, y_begin:y_end, x_begin:x_end].set(n_index**2)

            e_conv_all = to_conv_mat(cell, self.fourier_order)
            o_e_conv_all = to_conv_mat(1 / cell, self.fourier_order)

            de_ri, de_ti = self.solve(wl, e_conv_all, o_e_conv_all)

            self.spectrum_r = self.spectrum_r.at[i].set(de_ri)
            self.spectrum_t = self.spectrum_t.at[i].set(de_ti)

        return self.spectrum_r, self.spectrum_t

    def jax_test(self):
        # TODO
        # # Z  Y  X
        # # si = [[z_begin, z_end], [y_begin, y_end], [x_begin, x_end]]
        # if self.grating_type == 0:
        #     cell = np.ones((2, 1, 10))
        #     si = [3.48, 0, 1, 0, 1, 0, 3]
        #     ox = [3.48, 1, 2, 0, 1, 0, 3]
        # elif self.grating_type == 1:
        #     pass
        # elif self.grating_type == 2:
        #     cell = np.ones((2, 10, 10))
        #     si = [3.48, 0, 1, 0, 10, 0, 3]
        #     ox = [3.48, 1, 2, 0, 10, 0, 3]
        # else:
        #     raise ValueError
        #
        # for i, wl in enumerate(self.wls):
        #     for material, z_begin, z_end, y_begin, y_end, x_begin, x_end in [si, ox]:
        #         if material is str:
        #             n_index = find_n_index(material, wl)
        #         else:
        #             n_index = material
        #         cell[z_begin:z_end, y_begin:y_end, x_begin:x_end] = n_index ** 2

        for i, wl in enumerate(self.wls):
            e_conv_all = to_conv_mat(self.patterns, self.fourier_order)
            oneover_e_conv_all = to_conv_mat(1 / self.patterns, self.fourier_order)

            de_ri, de_ti = self.solve(wl, e_conv_all, oneover_e_conv_all)

            self.spectrum_r = de_ri
            self.spectrum_t = de_ti

        return self.spectrum_r, self.spectrum_t


if __name__ == '__main__':
    grating_type = 0
    pol = 0

    n_I = 1
    n_II = 1

    theta = 0
    phi = 0
    psi = 0 if pol else 90

    wls = jnp.linspace(500, 1300, 100)
    # wavelength = np.linspace(600, 800, 3)

    if grating_type in (0, 1):
        period = [700]
        patterns = [[3.48, 1, 0], [3.48, 1, 0]]  # n_ridge, n_groove, fill_factor
        fourier_order = 40

    elif grating_type == 2:
        period = [700, 700]
        patterns = [[3.48, 1, [0.3, 1]], [3.48, 1, [0.3, 1]]]  # n_ridge, n_groove, fill_factor[x, y]
        fourier_order = 2
    else:
        raise ValueError

    thickness = [460, 660]

    mode = 0  # 0: speed mode; 1: backprop mode;

    AA = RCWAOpt(grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
              fourier_order=fourier_order, wls=wls, period=period, patterns=patterns, thickness=thickness, mode=mode)
    t0 = time.perf_counter()

    a, b = AA.loop_wavelength_fill_factor()
    AA.plot()

    print(time.perf_counter() - t0)

    # AA.loop_wavelength_ucell()
    # AA.plot()
    # print('end')
