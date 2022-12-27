from .scattering_method import scattering_1d_1, scattering_1d_2, scattering_1d_3, scattering_2d_1, scattering_2d_wv,\
    scattering_2d_2, scattering_2d_3
from .transfer_method import transfer_1d_1, transfer_1d_2, transfer_1d_3, transfer_1d_conical_1, transfer_1d_conical_2,\
    transfer_1d_conical_3, transfer_2d_1, transfer_2d_wv, transfer_2d_2, transfer_2d_3

import meent.integ.backend.meentpy as ee


class _BaseRCWA:

    def __init__(self, grating_type, n_I=1., n_II=1., theta=0., phi=0., psi=0., fourier_order=10,
                 period=0.7, wavelength=900, pol=0,
                 patterns=None, ucell=None, ucell_materials=None, thickness=None, algo='TMM'):

        self.grating_type = grating_type  # 1D=0, 1D_conical=1, 2D=2
        self.n_I = n_I
        self.n_II = n_II

        self.theta = theta * ee.pi / 180
        self.phi = phi * ee.pi / 180
        self.psi = psi * ee.pi / 180  # TODO: integrate psi and pol

        self.pol = pol  # TE 0, TM 1
        if self.pol == 0:  # TE
            self.psi = 90 * ee.pi / 180
        elif self.pol == 1:  # TM
            self.psi = 0 * ee.pi / 180
        else:
            print('not implemented yet')
            raise ValueError

        self.fourier_order = fourier_order
        self.ff = 2 * self.fourier_order + 1

        self.period = period

        self.wavelength = wavelength

        self.patterns = patterns
        self.ucell = ucell
        self.ucell_materials = ucell_materials
        self.thickness = thickness

        self.algo = algo

        self.layer_info_list = []
        self.T1 = None

    def solve_1d(self, wl, E_conv_all, o_E_conv_all):

        fourier_indices = ee.arange(-self.fourier_order, self.fourier_order + 1)

        delta_i0 = ee.zeros(self.ff)

        # delta_i0[self.fourier_order] = 1
        delta_i0 = ee.assign(delta_i0, self.fourier_order, 1)

        k0 = 2 * ee.pi / wl

        if self.algo == 'TMM':
            kx_vector, Kx, k_I_z, k_II_z, f, YZ_I, g, inc_term, T \
                = transfer_1d_1(self.ff, self.pol, k0, self.n_I, self.n_II,
                                self.theta, delta_i0, self.fourier_order, fourier_indices, wl, self.period)
        elif self.algo == 'SMM':
            Kx, Wg, Vg, Kzg, Wr, Vr, Kzr, Wt, Vt, Kzt, Ar, Br, Sg \
                = scattering_1d_1(k0, self.n_I, self.n_II, self.theta, self.phi, fourier_indices, self.period,
                                  self.pol, wl=wl)
        else:
            raise ValueError

        # From the last layer
        for E_conv, o_E_conv, d in zip(E_conv_all[::-1], o_E_conv_all[::-1], self.thickness[::-1]):

            if self.pol == 0:
                E_conv_i = None
                A = Kx ** 2 - E_conv
                eigenvalues, W = ee.linalg.eig(A)
                q = eigenvalues ** 0.5

                Q = ee.diag(q)
                V = W @ Q

            elif self.pol == 1:
                E_conv_i = ee.linalg.inv(E_conv)
                B = Kx @ E_conv_i @ Kx - ee.eye(E_conv.shape[0])
                o_E_conv_i = ee.linalg.inv(o_E_conv)

                eigenvalues, W = ee.linalg.eig(o_E_conv_i @ B)
                q = eigenvalues ** 0.5

                Q = ee.diag(q)
                V = o_E_conv @ W @ Q

            else:
                raise ValueError

            if self.algo == 'TMM':
                X, f, g, T, a_i, b = transfer_1d_2(k0, q, d, W, V, f, g, self.fourier_order, T)

                layer_info = [E_conv_i, q, W, X, a_i, b, d]
                self.layer_info_list.append(layer_info)

            elif self.algo == 'SMM':
                A, B, S_dict, Sg = scattering_1d_2(W, Wg, V, Vg, d, k0, Q, Sg)
            else:
                raise ValueError

        if self.algo == 'TMM':
            de_ri, de_ti, T1 = transfer_1d_3(g, YZ_I, f, delta_i0, inc_term, T, k_I_z, k0, self.n_I, self.n_II,
                                         self.theta, self.pol, k_II_z)
            self.T1 = T1

        elif self.algo == 'SMM':
            de_ri, de_ti = scattering_1d_3(Wt, Wg, Vt, Vg, Sg, self.ff, Wr, self.fourier_order, Kzr, Kzt,
                                           self.n_I, self.n_II, self.theta, self.pol)
        else:
            raise ValueError

        return de_ri, de_ti

    # TODO: scattering method
    def solve_1d_conical(self, wl, e_conv_all, o_e_conv_all):

        fourier_indices = ee.arange(-self.fourier_order, self.fourier_order + 1)

        delta_i0 = ee.zeros(self.ff)
        delta_i0[self.fourier_order] = 1

        k0 = 2 * ee.pi / wl

        if self.algo == 'TMM':
            Kx, ky, k_I_z, k_II_z, varphi, Y_I, Y_II, Z_I, Z_II, big_F, big_G, big_T \
                = transfer_1d_conical_1(self.ff, k0, self.n_I, self.n_II, self.period, fourier_indices, self.theta, self.phi, wl)
        elif self.algo == 'SMM':
            print('SMM for 1D conical is not implemented')
            return ee.nan, ee.nan
        else:
            raise ValueError

        for e_conv, o_e_conv, d in zip(e_conv_all[::-1], o_e_conv_all[::-1], self.thickness[::-1]):
            e_conv_i = ee.linalg.inv(e_conv)
            o_e_conv_i = ee.linalg.inv(o_e_conv)

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

    def solve_2d(self, wl, E_conv_all, o_E_conv_all):

        fourier_indices = ee.arange(-self.fourier_order, self.fourier_order + 1)

        delta_i0 = ee.zeros((self.ff ** 2, 1))
        # delta_i0[self.ff ** 2 // 2, 0] = 1

        assign_index = [self.ff ** 2 // 2, 0]
        delta_i0 = ee.assign(delta_i0, assign_index, 1)

        I = ee.eye(self.ff ** 2)
        O = ee.zeros((self.ff ** 2, self.ff ** 2))

        center = self.ff ** 2

        k0 = 2 * ee.pi / wl

        if self.algo == 'TMM':
            kx_vector, ky_vector, Kx, Ky, k_I_z, k_II_z, varphi, Y_I, Y_II, Z_I, Z_II, big_F, big_G, big_T \
                = transfer_2d_1(self.ff, k0, self.n_I, self.n_II, self.period, fourier_indices, self.theta, self.phi, wl)
        elif self.algo == 'SMM':
            Kx, Ky, kz_inc, Wg, Vg, Kzg, Wr, Vr, Kzr, Wt, Vt, Kzt, Ar, Br, Sg \
                = scattering_2d_1(self.n_I, self.n_II, self.theta, self.phi, k0, self.period, self.fourier_order)
        else:
            raise ValueError

        # From the last layer
        for E_conv, o_E_conv, d in zip(E_conv_all[::-1], o_E_conv_all[::-1], self.thickness[::-1]):
            E_conv_i = ee.linalg.inv(E_conv)
            o_E_conv_i = ee.linalg.inv(o_E_conv)

            if self.algo == 'TMM':  # TODO: MERGE W V part
                W, V, q = transfer_2d_wv(self.ff, Kx, E_conv_i, Ky, o_E_conv_i, E_conv, center)

                big_X, big_F, big_G, big_T, big_A_i, big_B, \
                W_11, W_12, W_21, W_22, V_11, V_12, V_21, V_22 \
                    = transfer_2d_2(k0, d, W, V, center, q, varphi, I, O, big_F, big_G, big_T)

                layer_info = [E_conv_i, q, W_11, W_12, W_21, W_22, V_11, V_12, V_21, V_22, big_X, big_A_i, big_B, d]
                self.layer_info_list.append(layer_info)

            elif self.algo == 'SMM':
                W, V, LAMBDA = scattering_2d_wv(self.ff, Kx, Ky, E_conv, o_E_conv, o_E_conv_i, E_conv_i)
                A, B, Sl_dict, Sg_matrix, Sg = scattering_2d_2(W, Wg, V, Vg, d, k0, Sg, LAMBDA)
            else:
                raise ValueError

        if self.algo == 'TMM':
            de_ri, de_ti, big_T1 = transfer_2d_3(center, big_F, big_G, big_T, Z_I, Y_I, self.psi, self.theta, self.ff,
                                         delta_i0, k_I_z, k0, self.n_I, self.n_II, k_II_z)
            self.T1 = big_T1

        elif self.algo == 'SMM':
            de_ri, de_ti = scattering_2d_3(Wt, Wg, Vt, Vg, Sg, Wr, Kx, Ky, Kzr, Kzt, kz_inc, self.n_I,
                                           self.pol, self.theta, self.phi, self.fourier_order, self.ff)
        else:
            raise ValueError

        return de_ri.reshape((self.ff, self.ff)).real, de_ti.reshape((self.ff, self.ff)).real
