import numpy as np
import matplotlib.pyplot as plt

from .scattering_method import scattering_1d_1, scattering_1d_2, scattering_1d_3, scattering_2d_1, scattering_2d_wv,\
    scattering_2d_2, scattering_2d_3
from .transfer_method import transfer_1d_1, transfer_1d_2, transfer_1d_3, transfer_1d_conical_1, transfer_1d_conical_2,\
    transfer_1d_conical_3, transfer_2d_1, transfer_2d_wv, transfer_2d_2, transfer_2d_3


class Base:
    def __init__(self, grating_type):
        self.grating_type = grating_type
        self.wls = None
        self.fourier_order = None
        self.spectrum_r = None
        self.spectrum_t = None

    def init_spectrum_array(self):
        if self.grating_type in (0, 1):
            self.spectrum_r = np.zeros((len(self.wls), 2 * self.fourier_order + 1))
            self.spectrum_t = np.zeros((len(self.wls), 2 * self.fourier_order + 1))
        elif self.grating_type == 2:
            self.spectrum_r = np.zeros((len(self.wls), 2 * self.fourier_order + 1, 2 * self.fourier_order + 1))
            self.spectrum_t = np.zeros((len(self.wls), 2 * self.fourier_order + 1, 2 * self.fourier_order + 1))
        else:
            raise ValueError

    def save_spectrum_array(self, de_ri, de_ti, i):
        de_ri = np.array(de_ri)
        de_ti = np.array(de_ti)

        if not de_ri.shape:
            # 1D or may be not; there is a case that reticolo returns single value
            c = self.spectrum_r.shape[1] // 2
            self.spectrum_r[i][c] = de_ri

        elif len(de_ri.shape) == 1 or de_ri.shape[1] == 1:
            de_ri = de_ri.flatten()
            c = self.spectrum_r.shape[1] // 2
            l = de_ri.shape[0] // 2
            if len(de_ri) % 2:
                self.spectrum_r[i][c - l:c + l + 1] = de_ri
            else:
                self.spectrum_r[i][c - l:c + l] = de_ri

        else:
            print('no code')
            raise ValueError

        if not de_ti.shape:  # 1D
            c = self.spectrum_t.shape[1] // 2
            self.spectrum_t[i][c] = de_ti

        elif len(de_ti.shape) == 1 or de_ti.shape[1] == 1:  # 1D
            de_ti = de_ti.flatten()
            c = self.spectrum_t.shape[1] // 2
            l = de_ti.shape[0] // 2
            if len(de_ti) % 2:
                self.spectrum_t[i][c - l:c + l + 1] = de_ti
            else:
                self.spectrum_t[i][c - l:c + l] = de_ti

        else:
            print('no code')
            raise ValueError

    def plot(self, title=None, marker=None):
        if self.grating_type in (0, 1):
            plt.plot(self.wls, self.spectrum_r.sum(axis=1), marker=marker)
            plt.plot(self.wls, self.spectrum_t.sum(axis=1), marker=marker)
        elif self.grating_type == 2:
            plt.plot(self.wls, self.spectrum_r.sum(axis=(1, 2)), marker=marker)
            plt.plot(self.wls, self.spectrum_t.sum(axis=(1, 2)), marker=marker)
        else:
            raise ValueError
        plt.title(title)
        plt.show()


class _BaseRCWA(Base):
    def __init__(self, grating_type, n_I=1., n_II=1., theta=0., phi=0., psi=0., fourier_order=10,
                 period=0.7, wls=np.linspace(0.5, 2.3, 400), pol=0,
                 patterns=None, ucell=None, ucell_materials=None, thickness=None, algo='TMM'):
        super().__init__(grating_type)

        self.grating_type = grating_type  # 1D=0, 1D_conical=1, 2D=2
        self.n_I = n_I
        self.n_II = n_II

        self.theta = theta * np.pi / 180
        self.phi = phi * np.pi / 180
        self.psi = psi * np.pi / 180  # TODO: integrate psi and pol

        self.pol = pol  # TE 0, TM 1
        if self.pol == 0:  # TE
            self.psi = 90 * np.pi / 180
        elif self.pol == 1:  # TM
            self.psi = 0 * np.pi / 180
        else:
            print('not implemented yet')
            raise ValueError

        self.fourier_order = fourier_order
        self.ff = 2 * self.fourier_order + 1

        self.period = period

        self.wls = wls

        self.patterns = patterns
        self.ucell = ucell
        self.ucell_materials = ucell_materials
        self.thickness = thickness

        self.algo = algo

        self.init_spectrum_array()

    def solve_1d(self, wl, E_conv_all, oneover_E_conv_all):

        fourier_indices = np.arange(-self.fourier_order, self.fourier_order + 1)

        delta_i0 = np.zeros(self.ff)
        delta_i0[self.fourier_order] = 1

        k0 = 2 * np.pi / wl

        # --------------------------------------------------------------------
        if self.algo == 'TMM':
            Kx, k_I_z, k_II_z, Kx, f, YZ_I, g, inc_term, T \
                = transfer_1d_1(self.ff, self.pol, k0, self.n_I, self.n_II,
                                self.theta, delta_i0, self.fourier_order, fourier_indices, wl, self.period)
        elif self.algo == 'SMM':
            Kx, Wg, Vg, Kzg, Wr, Vr, Kzr, Wt, Vt, Kzt, Ar, Br, Sg \
                = scattering_1d_1(k0, self.n_I, self.n_II, self.theta, self.phi, fourier_indices, self.period,
                                  self.pol, wl=wl)
        else:
            raise ValueError

        # --------------------------------------------------------------------
        for E_conv, oneover_E_conv, d in zip(E_conv_all[::-1], oneover_E_conv_all[::-1], self.thickness[::-1]):
            if self.pol == 0:
                A = Kx ** 2 - E_conv
                eigenvalues, W = np.linalg.eig(A)
                q = eigenvalues ** 0.5

                Q = np.diag(q)
                V = W @ Q

            elif self.pol == 1:
                E_i = np.linalg.inv(E_conv)
                B = Kx @ E_i @ Kx - np.eye(E_conv.shape[0])
                oneover_E_conv_i = np.linalg.inv(oneover_E_conv)

                eigenvalues, W = np.linalg.eig(oneover_E_conv_i @ B)
                q = eigenvalues ** 0.5

                Q = np.diag(q)
                V = oneover_E_conv @ W @ Q

            else:
                raise ValueError
            # --------------------------------------------------------------------
            if self.algo == 'TMM':
                f, g, T = transfer_1d_2(k0, q, d, W, V, f, g, self.fourier_order, T)
            elif self.algo == 'SMM':
                A, B, S_dict, Sg = scattering_1d_2(W, Wg, V, Vg, d, k0, Q, Sg)
            else:
                raise ValueError

        if self.algo == 'TMM':
            de_ri, de_ti = transfer_1d_3(g, YZ_I, f, delta_i0, inc_term, T, k_I_z, k0, self.n_I, self.n_II,
                                         self.theta, self.pol, k_II_z)
        elif self.algo == 'SMM':
            de_ri, de_ti = scattering_1d_3(Wt, Wg, Vt, Vg, Sg, self.ff, Wr, self.fourier_order, Kzr, Kzt,
                                           self.n_I, self.n_II, self.theta, self.pol)
        else:
            raise ValueError

        return de_ri, de_ti

    # TODO: scattering method
    def solve_1d_conical(self, wl, e_conv_all, o_e_conv_all):

        fourier_indices = np.arange(-self.fourier_order, self.fourier_order + 1)

        delta_i0 = np.zeros(self.ff)
        delta_i0[self.fourier_order] = 1

        k0 = 2 * np.pi / wl

        if self.algo == 'TMM':
            Kx, ky, k_I_z, k_II_z, varphi, Y_I, Y_II, Z_I, Z_II, big_F, big_G, big_T \
                = transfer_1d_conical_1(self.ff, k0, self.n_I, self.n_II, self.period, fourier_indices, self.theta, self.phi, wl)
        elif self.algo == 'SMM':
            print('SMM for 1D conical is not implemented')
            return np.nan, np.nan
        else:
            raise ValueError

        for e_conv, o_e_conv, d in zip(e_conv_all[::-1], o_e_conv_all[::-1], self.thickness[::-1]):
            e_conv_i = np.linalg.inv(e_conv)
            o_e_conv_i = np.linalg.inv(o_e_conv)

            if self.algo == 'TMM':
                big_F, big_G, big_T = transfer_1d_conical_2(k0, Kx, ky, e_conv, e_conv_i, o_e_conv_i, self.ff, d,
                                                            varphi, big_F, big_G, big_T)
            elif self.algo == 'SMM':
                raise ValueError
            else:
                raise ValueError

        if self.algo == 'TMM':
             de_ri, de_ti = transfer_1d_conical_3(big_F, big_G, big_T, Z_I, Y_I, self.psi, self.theta, self.ff,
                                                  delta_i0, k_I_z, k0, self.n_I, self.n_II, k_II_z)
        elif self.algo == 'SMM':
            raise ValueError
        else:
            raise ValueError

        return de_ri, de_ti

    def solve_2d(self, wl, E_conv_all, oneover_E_conv_all):

        fourier_indices = np.arange(-self.fourier_order, self.fourier_order + 1)

        delta_i0 = np.zeros((self.ff ** 2, 1))
        delta_i0[self.ff ** 2 // 2, 0] = 1

        I = np.eye(self.ff ** 2)
        O = np.zeros((self.ff ** 2, self.ff ** 2))

        center = self.ff ** 2

        k0 = 2 * np.pi / wl

        if self.algo == 'TMM':
            Kx, Ky, k_I_z, k_II_z, varphi, Y_I, Y_II, Z_I, Z_II, big_F, big_G, big_T \
                = transfer_2d_1(self.ff, k0, self.n_I, self.n_II, self.period, fourier_indices, self.theta, self.phi, wl)
        elif self.algo == 'SMM':
            Kx, Ky, kz_inc, Wg, Vg, Kzg, Wr, Vr, Kzr, Wt, Vt, Kzt, Ar, Br, Sg \
                = scattering_2d_1(self.n_I, self.n_II, self.theta, self.phi, k0, self.period, self.fourier_order)
        else:
            raise ValueError

        for E_conv, oneover_E_conv, d in zip(E_conv_all[::-1], oneover_E_conv_all[::-1], self.thickness[::-1]):
            E_i = np.linalg.inv(E_conv)
            oneover_E_conv_i = np.linalg.inv(oneover_E_conv)

            if self.algo == 'TMM':  # TODO: MERGE W V part
                W, V, LAMBDA, Lambda = transfer_2d_wv(self.ff, Kx, E_i, Ky, oneover_E_conv_i, E_conv, center)
                big_F, big_G, big_T = transfer_2d_2(k0, d, W, V, center, Lambda, varphi, I, O, big_F, big_G, big_T)
            elif self.algo == 'SMM':
                W, V, LAMBDA = scattering_2d_wv(self.ff, Kx, Ky, E_conv, oneover_E_conv, oneover_E_conv_i, E_i)
                A, B, Sl_dict, Sg_matrix, Sg = scattering_2d_2(W, Wg, V, Vg, d, k0, Sg, LAMBDA)
            else:
                raise ValueError

        if self.algo == 'TMM':
            de_ri, de_ti = transfer_2d_3(center, big_F, big_G, big_T, Z_I, Y_I, self.psi, self.theta, self.ff,
                                         delta_i0, k_I_z, k0, self.n_I, self.n_II, k_II_z)
        elif self.algo == 'SMM':
            de_ri, de_ti = scattering_2d_3(Wt, Wg, Vt, Vg, Sg, Wr, Kx, Ky, Kzr, Kzt, kz_inc, self.n_I,
                                           self.pol, self.theta, self.phi, self.fourier_order, self.ff)
        else:
            raise ValueError

        return de_ri.reshape((self.ff, self.ff)).real, de_ti.reshape((self.ff, self.ff)).real
