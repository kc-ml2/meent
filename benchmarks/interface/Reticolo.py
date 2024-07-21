import os
import numpy as np

import meent

from meent.on_numpy.modeler.modeling import find_nk_index


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

    def run_res2(self, grating_type, period, fourier_order, ucell, thickness, theta, phi, pol, wavelength, n_I, n_II,
                 *args, **kwargs):
        phi *= (180 / np.pi)

        if grating_type in (0, 1):
            period = period[0]

            fourier_order = fourier_order
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
            self._run(pol, theta, phi, period, n_I, fourier_order, textures, profile, wavelength, grating_type,
                      cal_field=False)

        return top_refl_info.efficiency, top_tran_info.efficiency, bottom_refl_info.efficiency, bottom_tran_info.efficiency

    def run_res3(self, grating_type, period, fourier_order, ucell, thickness, theta, phi, pol, wavelength, n_I, n_II,
                 matlab_plot_field=0, res3_npts=0, *args, **kwargs):
        phi *= (180 / np.pi)

        if grating_type in (0, 1):
            period = period[0]

            fourier_order = fourier_order
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

        top_refl_info, top_tran_info, bottom_refl_info, bottom_tran_info, field_cell = \
            self._run(pol, theta, phi, period, n_I, fourier_order, textures, profile, wavelength, grating_type,
                      cal_field=True, matlab_plot_field=matlab_plot_field, res3_npts=res3_npts)

        return top_refl_info.efficiency, top_tran_info.efficiency, bottom_refl_info.efficiency, \
               bottom_tran_info.efficiency, field_cell

    def _run(self, pol, theta, phi, period, n_I, fourier_order,
             textures, profile, wavelength, grating_type, cal_field=False, matlab_plot_field=0, res3_npts=0):

        if cal_field:
            top_refl_info, top_tran_info, bottom_refl_info, bottom_tran_info, field_cell = \
                self.eng.reticolo_res3(pol, theta, phi, period, n_I, fourier_order,
                                       textures, profile, wavelength, grating_type, matlab_plot_field, res3_npts,
                                       nout=5)
            res = (top_refl_info, top_tran_info, bottom_refl_info, bottom_tran_info, field_cell)
        else:
            top_refl_info, top_tran_info, bottom_refl_info, bottom_tran_info = \
                self.eng.reticolo_res2(pol, theta, phi, period, n_I, fourier_order,
                                       textures, profile, wavelength, grating_type, nout=4)
            res = (top_refl_info, top_tran_info, bottom_refl_info, bottom_tran_info)
        return res

    # def run_acs(self, pattern, n_si='SILICON'):
    #     if type(n_si) == str and n_si.upper() == 'SILICON':
    #         n_si = find_nk_index(n_si, self.mat_table, self.wavelength)
    #
    #     abseff, effi_r, effi_t = self.eng.reticolo_res2(pattern, self.wavelength, self.deflected_angle,
    #                                                     self.fourier_order,
    #                                                     self.n_top, self.n_bot, self.thickness, self.theta, n_si, nout=3)
    #     effi_r, effi_t = np.array(effi_r).flatten(), np.array(effi_t).flatten()
    #
    #     return abseff, effi_r, effi_t


if __name__ == '__main__':

    option = {
        'grating_type': 1,
        'pol': 1,
        'n_top': 1,
        'n_bot': 1,
        'theta': 1,
        'phi': 1,
        'wavelength': 1,
        'fourier_order': 1,
        'thickness': [1000, 300],
        'period': [1000],
        'fourier_type': 1,
        'ucell': np.array(
            [
                [[3.1, 1.1, 1.2, 1.6, 3.1]],
                [[3, 3, 1, 1, 1]],
            ]
        ),
    }

    reti = Reticolo()
    reti_de_ri, reti_de_ti, c, d, r_field_cell = reti.run_res3(**option)
    print('reti de_ri', np.array(reti_de_ri).flatten())
    print('reti de_ti', np.array(reti_de_ti).flatten())

    # Numpy
    backend = 0
    nmee = meent.call_mee(backend=backend, perturbation=1E-30, **option)
    n_de_ri, n_de_ti = nmee.conv_solve()
    n_field_cell = nmee.calculate_field(res_z=200, res_x=5)
    print('nmeent de_ri', n_de_ri)
    print('nmeent de_ti', n_de_ti)

    # JAX
    backend = 1
    jmee = meent.call_mee(backend=backend, perturbation=1E-30, **option)
    j_de_ri, j_de_ti = jmee.conv_solve()
    j_field_cell = jmee.calculate_field(res_z=200, res_x=5)
    print('jmeent de_ri', j_de_ri)
    print('jmeent de_ti', j_de_ti)

    # Torch
    backend = 2
    tmee = meent.call_mee(backend=backend, perturbation=1E-30, **option)
    t_de_ri, t_de_ti = tmee.conv_solve()
    t_field_cell = tmee.calculate_field(res_z=200, res_x=5)
    print('tmeent de_ri', t_de_ri)
    print('tmeent de_ti', t_de_ti)

    import matplotlib.pyplot as plt

    plt.imshow(abs(r_field_cell[:,:,0]**2), cmap='jet', aspect='auto')
    plt.colorbar()
    plt.show()

    plt.imshow(abs(n_field_cell[:,0,:,0]**2), cmap='jet', aspect='auto')
    plt.colorbar()
    plt.show()

    plt.imshow(abs(j_field_cell[:,0,:,0]**2), cmap='jet', aspect='auto')
    plt.colorbar()
    plt.show()

    plt.imshow(abs(t_field_cell[:,0,:,0]**2), cmap='jet', aspect='auto')
    plt.colorbar()
    plt.show()
