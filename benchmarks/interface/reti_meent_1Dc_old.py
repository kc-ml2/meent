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

    option = {}
    option['grating_type'] = 1  # 0 : just 1D grating, 1 : 1D rotating grating, 2 : 2D grating
    option['pol'] = 1  # 0: TE, 1: TM
    option['n_top'] = 1  # n_incidence
    option['n_bot'] = 1.5  # n_transmission
    option['theta'] = 30 * np.pi / 180
    option['phi'] = 0 * np.pi / 180
    option['fto'] = 40
    option['period'] = [1000]
    option['wavelength'] = 650
    option['thickness'] = [100, 100, 100, 100, 100, 100]  # final term is for h_substrate
    option['fourier_type'] = 2

    ucell = np.array(
        [
            [[1, 1, 1, 1, 1, 0, 0, 1, 1, 1, ]],
            [[1, 0, 0, 1, 0, 0, 0, 1, 1, 1, ]],
            [[1, 1, 0, 1, 1, 1, 1, 1, 0, 1, ]],
            [[1, 1, 1, 0, 1, 0, 0, 1, 1, 1, ]],
            [[0, 0, 1, 0, 1, 0, 0, 1, 1, 1, ]],
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]],
        ]) * (3 - 1) + 1
    ucell = np.repeat(ucell, 10, axis=2)
    option['ucell'] = ucell

    res3_npts = 20
    reti = Reticolo()
    reti_de_ri, reti_de_ti, c, d, r_field_cell = reti.run_res3(**option, res3_npts=res3_npts)
    print('reti de_ri', np.array(reti_de_ri).flatten())
    print('reti de_ti', np.array(reti_de_ti).flatten())

    # Numpy
    backend = 0
    nmee = meent.call_mee(backend=backend, **option)
    n_de_ri, n_de_ti = nmee.conv_solve()
    n_field_cell = nmee.calculate_field(res_z=res3_npts, res_x=100)
    print('nmeent de_ri', n_de_ri[n_de_ri > 1E-5])
    print('nmeent de_ti', n_de_ti[n_de_ti > 1E-5])

    if True:
        a0 = np.flipud(r_field_cell[res3_npts:-res3_npts, :, 0])
        b0 = n_field_cell[:, 0, :, 0]
        print('Ex', np.linalg.norm(a0.conj() - b0))

        a1 = np.flipud(r_field_cell[res3_npts:-res3_npts, :, 1])
        b1 = n_field_cell[:, 0, :, 1]
        print('Ey', np.linalg.norm(a1.conj() - b1))

        a2 = np.flipud(r_field_cell[res3_npts:-res3_npts, :, 2])
        b2 = n_field_cell[:, 0, :, 2]
        print('Ez', np.linalg.norm(a2.conj() - b2))

        a3 = np.flipud(r_field_cell[res3_npts:-res3_npts, :, 3])
        b3 = n_field_cell[:, 0, :, 3]
        print('Hx', np.linalg.norm(a3.conj() - b3))

        a4 = np.flipud(r_field_cell[res3_npts:-res3_npts, :, 4])
        b4 = n_field_cell[:, 0, :, 4]
        print('Hy', np.linalg.norm(a4.conj() - b4))

        a5 = np.flipud(r_field_cell[res3_npts:-res3_npts, :, 5])
        b5 = n_field_cell[:, 0, :, 5]
        print('Hz', np.linalg.norm(a5.conj() - b5))

    title = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz', ]
    fig, axes = plt.subplots(2, 6, figsize=(15, 4))

    for ix in range(len(title)):
        data = r_field_cell[res3_npts:-res3_npts, :, ix]
        val = data.real
        # val = np.clip(val, -1, 1)
        im = axes[0, ix].imshow(np.flipud(val), cmap='jet', aspect='auto')
        fig.colorbar(im, ax=axes[0, ix], shrink=1)
        axes[0, ix].title.set_text(title[ix])

        val = data.imag
        im = axes[1, ix].imshow(np.flipud(val), cmap='jet', aspect='auto')
        fig.colorbar(im, ax=axes[1, ix], shrink=1)
    plt.show()

    fig, axes = plt.subplots(2, 6, figsize=(15, 4))

    for ix in range(len(title)):

        data = n_field_cell[:, 0, :, ix]
        val = data.real
        # val = np.clip(val, -1, 1)
        im = axes[0, ix].imshow(val, cmap='jet', aspect='auto')
        fig.colorbar(im, ax=axes[0, ix], shrink=1)
        axes[0, ix].title.set_text(title[ix])

        val = -data.imag
        # val = np.clip(val, -1, 1)
        im = axes[1, ix].imshow(val, cmap='jet', aspect='auto')
        fig.colorbar(im, ax=axes[1, ix], shrink=1)

    plt.show()

    print('End')
