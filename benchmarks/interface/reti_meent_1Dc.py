import os
import numpy as np
import matplotlib.pyplot as plt

import meent

# os.environ['OCTAVE_EXECUTABLE'] = '/opt/homebrew/bin/octave-cli'


class Reticolo:

    def __init__(self, engine_type='octave', *args, **kwargs):

        if engine_type == 'octave':
            try:
                from oct2py import octave
            except Exception as e:
                raise e
            self.eng = octave

        elif engine_type == 'matlab':
            try:
                import matlab.engine
            except Exception as e:
                raise e
            self.eng = matlab.engine.start_matlab()
        else:
            raise ValueError

        # path that has file to run in octave
        m_path = os.path.dirname(__file__)
        self.eng.addpath(self.eng.genpath(m_path))

    def run_res2(self, grating_type, period, fto, ucell, thickness, theta, phi, pol, wavelength, n_I, n_II,
                 *args, **kwargs):
        theta *= (180 / np.pi)
        phi *= (180 / np.pi)

        if grating_type in (0, 1):
            period = period[0]

            fto = fto
            Nx = ucell.shape[2]
            period_x = period
            grid_x = np.linspace(0, period, Nx + 1)[1:]
            grid_x -= period_x / 2

            # grid = np.linspace(0, period, Nx)

            ucell_new = []
            for z in range(ucell.shape[0]):
                ucell_layer = [grid_x, ucell[z, 0]]
                ucell_new.append(ucell_layer)

            textures = [n_I, *ucell_new, n_II]

        else:

            Nx = ucell.shape[2]
            Ny = ucell.shape[1]
            period_x = period[0]
            period_y = period[1]

            unit_x = period_x / Nx
            unit_y = period_y / Ny

            grid_x = np.linspace(0, period[0], Nx + 1)[1:]
            grid_y = np.linspace(0, period[1], Ny + 1)[1:]

            grid_x -= period_x / 2
            grid_y -= period_y / 2

            ucell_new = []
            for z in range(ucell.shape[0]):
                ucell_layer = [10]
                for y, yval in enumerate(grid_y):
                    for x, xval in enumerate(grid_x):
                        obj = [xval, yval, unit_x, unit_y, ucell[z, y, x], 1]
                        ucell_layer.append(obj)
                ucell_new.append(ucell_layer)
            textures = [n_I, *ucell_new, n_II]

        profile = np.array([[0, *thickness, 0], range(1, len(thickness) + 3)])

        top_refl_info, top_tran_info, bottom_refl_info, bottom_tran_info = \
            self._run(pol, theta, phi, period, n_I, fto, textures, profile, wavelength, grating_type,
                      cal_field=False)

        return top_refl_info.efficiency, top_tran_info.efficiency, bottom_refl_info.efficiency, bottom_tran_info.efficiency

    def run_res3(self, grating_type, period, fto, ucell, thickness, theta, phi, pol, wavelength, n_top, n_bot,
                 matlab_plot_field=0, res3_npts=0, *args, **kwargs):

        # theta *= (180 / np.pi)
        phi *= (180 / np.pi)

        if grating_type in (0, 1):
            period = period[0]

            fto = fto
            Nx = ucell.shape[2]
            period_x = period
            grid_x = np.linspace(0, period, Nx + 1)[1:]
            grid_x -= period_x / 2

            # grid = np.linspace(0, period, Nx)

            ucell_new = []
            for z in range(ucell.shape[0]):
                ucell_layer = [grid_x, ucell[z, 0]]
                ucell_new.append(ucell_layer)

            textures = [n_top, *ucell_new, n_bot]

        else:

            Nx = ucell.shape[2]
            Ny = ucell.shape[1]
            period_x = period[0]
            period_y = period[1]

            unit_x = period_x / Nx
            unit_y = period_y / Ny

            grid_x = np.linspace(0, period[0], Nx + 1)[1:]
            grid_y = np.linspace(0, period[1], Ny + 1)[1:]

            grid_x -= period_x / 2
            grid_y -= period_y / 2

            ucell_new = []
            for z in range(ucell.shape[0]):
                ucell_layer = [10]
                for y, yval in enumerate(grid_y):
                    for x, xval in enumerate(grid_x):
                        obj = [xval, yval, unit_x, unit_y, ucell[z, y, x], 1]
                        ucell_layer.append(obj)
                ucell_new.append(ucell_layer)
            textures = [n_top, *ucell_new, n_bot]

        profile = np.array([[0, *thickness, 0], range(1, len(thickness) + 3)])

        top_refl_info, top_tran_info, bottom_refl_info, bottom_tran_info, field_cell = \
            self._run(pol, theta, phi, period, n_top, fto, textures, profile, wavelength, grating_type,
                      cal_field=True, matlab_plot_field=matlab_plot_field, res3_npts=res3_npts)

        return top_refl_info.efficiency, top_tran_info.efficiency, bottom_refl_info.efficiency, \
               bottom_tran_info.efficiency, field_cell

    def _run(self, pol, theta, phi, period, n_top, fto,
             textures, profile, wavelength, grating_type, cal_field=False, matlab_plot_field=0, res3_npts=0):

        if cal_field:
            top_refl_info, top_tran_info, bottom_refl_info, bottom_tran_info, field_cell = \
                self.eng.reticolo_res3(pol, theta, phi, period, n_top, fto,
                                       textures, profile, wavelength, grating_type, matlab_plot_field, res3_npts,
                                       nout=5)
            res = (top_refl_info, top_tran_info, bottom_refl_info, bottom_tran_info, field_cell)
        else:
            top_refl_info, top_tran_info, bottom_refl_info, bottom_tran_info = \
                self.eng.reticolo_res2(pol, theta, phi, period, n_top, fto,
                                       textures, profile, wavelength, grating_type, nout=4)
            res = (top_refl_info, top_tran_info, bottom_refl_info, bottom_tran_info)
        return res


if __name__ == '__main__':

    factor = 100
    option = {}
    option['grating_type'] = 1  # 0 : just 1D grating, 1 : 1D rotating grating, 2 : 2D grating
    option['pol'] = 0  # 0: TE, 1: TM
    option['n_top'] = 2.2  # n_incidence
    option['n_bot'] = 2  # n_transmission
    option['theta'] = 40 * np.pi / 180
    option['phi'] = 20 * np.pi / 180
    option['fto'] = [40, 0]
    option['period'] = [770/factor]
    option['wavelength'] = 777/factor
    option['thickness'] = [100/factor, 100/factor, 100/factor, 100/factor, 100/factor, 100/factor]  # final term is for h_substrate
    option['thickness'] = [100/factor,]  # final term is for h_substrate
    option['fourier_type'] = 2

    ucell = np.array(
        [
            [[3, 3, 3, 3, 3, 1, 1, 1, 1, 1]],
        ])

    option['ucell'] = ucell

    res3_npts = 20
    reti = Reticolo()
    reti_de_ri, reti_de_ti, c, d, r_field_cell = reti.run_res3(**option, matlab_plot_field=0, res3_npts=res3_npts)
    print('reti de_ri', np.array(reti_de_ri).flatten())
    print('reti de_ti', np.array(reti_de_ti).flatten())

    # Numpy
    backend = 0
    nmee = meent.call_mee(backend=backend, perturbation=1E-30, **option)
    n_de_ri, n_de_ti = nmee.conv_solve()
    n_field_cell = nmee.calculate_field(res_z=20, res_x=ucell.shape[-1])

    # n_field_cell = np.roll(n_field_cell, -1, 2)

    print('nmeent de_ri', n_de_ri[n_de_ri > 1E-5])
    print('nmeent de_ti', n_de_ti[n_de_ti > 1E-5])


    if option['pol'] == 0:  # TE
        title = ['1D Ey', '1D Hx', '1D Hz', ]
    else:  # TM
        title = ['1D Hy', '1D Ex', '1D Ez', ]

    title = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']
    for i in range(len(title)):
        a0 = np.flipud(r_field_cell[res3_npts:-res3_npts, :, i])
        b0 = n_field_cell[:, 0, :, i]

        res = []
        res.append(np.linalg.norm(a0.conj() - b0).round(3))
        res.append(np.linalg.norm(abs(a0.conj())**2 - abs(b0)**2).round(3))
        res.append(np.linalg.norm(a0.conj().real - b0.real).round(3))
        res.append(np.linalg.norm(a0.conj().imag - b0.imag).round(3))

        print(f'{title[i]}, {res}')

        aa = np.angle(a0.conj())
        bb = np.angle(b0)

        print(aa[0][1:] - aa[0][:-1])
        print(bb[0][1:] - bb[0][:-1])

        print(aa[0] - bb[0])
        print(1)

    #
    # print('Ey, val diff', np.linalg.norm(a0.conj() - b0))
    # print('Ey, abs2 diff', np.linalg.norm(abs(a0.conj())**2 - abs(b0)**2))
    # print('Ey, real diff', np.linalg.norm(a0.conj().real - b0.real))
    # print('Ey, imag diff', np.linalg.norm(a0.conj().imag - b0.imag))
    #
    # a1 = np.flipud(r_field_cell[res3_npts:-res3_npts, :, 1])
    # b1 = n_field_cell[:, 0, :, 1]
    # print(np.linalg.norm(a1.conj() - b1))
    # print('Hx, val diff', np.linalg.norm(a1.conj() - b1))
    # print('Ey, abs2 diff', np.linalg.norm(abs(a1.conj())**2 - abs(b1)**2))
    # print('Ey, real diff', np.linalg.norm(a1.conj().real - b1.real))
    # print('Ey, imag diff', np.linalg.norm(a1.conj().imag - b1.imag))
    #
    # a2 = np.flipud(r_field_cell[res3_npts:-res3_npts, :, 2])
    # b2 = n_field_cell[:, 0, :, 2]
    # print(np.linalg.norm(a2.conj() - b2))
    # print('Hz, val diff', np.linalg.norm(a2.conj() - b2))
    # print('Ey, abs2 diff', np.linalg.norm(abs(a2.conj())**2 - abs(b2)**2))
    # print('Ey, real diff', np.linalg.norm(a2.conj().real - b2.real))
    # print('Ey, imag diff', np.linalg.norm(a2.conj().imag - b2.imag))

    fig, axes = plt.subplots(6, 6, figsize=(10, 5))

    for ix in range(len(title)):
        r_data = np.flipud(r_field_cell[res3_npts:-res3_npts, :, ix]).conj()

        im = axes[ix, 0].imshow(abs(r_data)**2, cmap='jet', aspect='auto')
        fig.colorbar(im, ax=axes[ix, 0], shrink=1)
        im = axes[ix, 2].imshow(r_data.real, cmap='jet', aspect='auto')
        fig.colorbar(im, ax=axes[ix, 2], shrink=1)
        im = axes[ix, 4].imshow(r_data.imag, cmap='jet', aspect='auto')
        fig.colorbar(im, ax=axes[ix, 4], shrink=1)


        n_data = n_field_cell[:, 0, :, ix]

        im = axes[ix, 1].imshow(abs(n_data)**2, cmap='jet', aspect='auto')
        fig.colorbar(im, ax=axes[ix, 1], shrink=1)

        im = axes[ix, 3].imshow(n_data.real, cmap='jet', aspect='auto')
        fig.colorbar(im, ax=axes[ix, 3], shrink=1)

        im = axes[ix, 5].imshow(n_data.imag, cmap='jet', aspect='auto')
        fig.colorbar(im, ax=axes[ix, 5], shrink=1)

    ix=0
    axes[ix, 0].title.set_text('abs**2 reti')
    axes[ix, 2].title.set_text('Re, reti')
    axes[ix, 4].title.set_text('Im, reti')
    axes[ix, 1].title.set_text('abs**2 meen')
    axes[ix, 3].title.set_text('Re, meen')
    axes[ix, 5].title.set_text('Im, meen')

    plt.show()

    1
