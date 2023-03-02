import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import meent

from meent.on_numpy.emsolver.convolution_matrix import find_nk_index

os.environ['OCTAVE_EXECUTABLE'] = '/opt/homebrew/bin/octave-cli'


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

    def run(self, period, fourier_order, ucell, thickness, theta, phi, pol, wavelength, n_I, n_II, *args, **kwargs):
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
            fourier_order = [fourier_order, fourier_order]

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

        top_refl_info, top_tran_info, bottom_refl_info, bottom_tran_info =\
            self._run(pol, theta, phi, period, n_I, fourier_order, textures, profile, wavelength, grating_type,)

        return top_refl_info.efficiency, top_tran_info.efficiency, bottom_refl_info.efficiency, bottom_tran_info.efficiency

    def _run(self, pol, theta, phi, period, n_I, fourier_order,
                                  textures, profile, wavelength, grating_type):

        top_refl_info, top_tran_info, bottom_refl_info, bottom_tran_info =\
            self.eng.run_reticolo(pol, theta, phi, period, n_I, fourier_order,
                                  textures, profile, wavelength, grating_type, nout=4)

        return top_refl_info, top_tran_info, bottom_refl_info, bottom_tran_info

    def run_acs(self, pattern, n_si='SILICON'):
        if type(n_si) == str and n_si.upper() == 'SILICON':
            n_si = find_nk_index(n_si, self.mat_table, self.wavelength)

        abseff, effi_r, effi_t = self.eng.Eval_Eff_1D(pattern, self.wavelength, self.deflected_angle, self.fourier_order,
                                                      self.n_I, self.n_II, self.thickness, self.theta, n_si, nout=3)
        effi_r, effi_t = np.array(effi_r).flatten(), np.array(effi_t).flatten()

        return abseff, effi_r, effi_t


if __name__ == '__main__':
    from meent.testcase import load_setting
    mode = 0
    dtype = 0
    device = 0
    grating_type = 2
    pre = load_setting(mode, dtype, device, grating_type)

    reti = Reticolo()
    a,b,c,d = reti.run(**pre)

    print(np.array(a).flatten()[::-1])
    print(np.array(b).flatten()[::-1])
    # print(np.array(a))
    # print(np.array(b))
    # print(c)
    # print(d)

    # Numpy
    mode = 0
    pre = load_setting(mode, dtype, device, grating_type)
    solver = meent.call_solver(mode=mode, perturbation=1E-30, **pre)
    solver.ucell = solver.ucell ** 2

    from meent.on_numpy.emsolver.convolution_matrix import to_conv_mat_discrete, to_conv_mat_continuous
    E_conv_all = to_conv_mat_continuous(solver.ucell, solver.fourier_order)
    o_E_conv_all = to_conv_mat_continuous(1 / solver.ucell, solver.fourier_order)
    # E_conv_all = to_conv_mat_discrete(solver.ucell, solver.fourier_order)
    # o_E_conv_all = to_conv_mat_discrete(1 / solver.ucell, solver.fourier_order)

    de_ri, de_ti, _, _, _ = solver.solve(solver.wavelength, E_conv_all, o_E_conv_all)
    c = de_ri.shape[0]//2
    try:
        print(de_ri[c-1:c+2, c-1:c+2])
        print(de_ti[c-1:c+2, c-1:c+2])
    except:
        # print(de_ri[c-1:c+2])
        # print(de_ti[c-1:c+2])
        print(de_ri)
        print(de_ti)

    print(a.sum(),de_ri.sum())
    print(b.sum(),de_ti.sum())
    print(a.sum()+b.sum(),de_ri.sum()+de_ti.sum())

    # JAX
    mode = 1
    pre = load_setting(mode, dtype, device, grating_type)
    solver = meent.call_solver(mode=mode, perturbation=1E-30, **pre)
    solver.ucell = solver.ucell ** 2

    from meent.on_jax.emsolver.convolution_matrix import to_conv_mat_discrete, to_conv_mat_continuous
    E_conv_all = to_conv_mat_continuous(solver.ucell, solver.fourier_order)
    o_E_conv_all = to_conv_mat_continuous(1 / solver.ucell, solver.fourier_order)
    # E_conv_all = to_conv_mat_discrete(solver.ucell, solver.fourier_order)
    # o_E_conv_all = to_conv_mat_discrete(1 / solver.ucell, solver.fourier_order)

    de_ri, de_ti, _, _, _ = solver.solve(solver.wavelength, E_conv_all, o_E_conv_all)
    c = de_ri.shape[0]//2
    try:
        print(de_ri[c-1:c+2,c-1:c+2])
        print(de_ti[c-1:c+2,c-1:c+2])
    except:
        # print(de_ri[c-1:c+2])
        # print(de_ti[c-1:c+2])
        print(de_ri)
        print(de_ti)

    print(a.sum(),de_ri.sum())
    print(b.sum(),de_ti.sum())
    print(a.sum()+b.sum(),de_ri.sum()+de_ti.sum())

    # Torch
    mode = 2
    pre = load_setting(mode, dtype, device, grating_type)
    solver = meent.call_solver(mode=mode, perturbation=1E-30, **pre)
    solver.ucell = solver.ucell ** 2

    from meent.on_torch.emsolver.convolution_matrix import to_conv_mat_discrete, to_conv_mat_continuous
    E_conv_all = to_conv_mat_continuous(solver.ucell, solver.fourier_order)
    o_E_conv_all = to_conv_mat_continuous(1 / solver.ucell, solver.fourier_order)
    # E_conv_all = to_conv_mat_discrete(solver.ucell, solver.fourier_order)
    # o_E_conv_all = to_conv_mat_discrete(1 / solver.ucell, solver.fourier_order)

    de_ri, de_ti, _, _, _ = solver.solve(solver.wavelength, E_conv_all, o_E_conv_all)
    c = de_ri.shape[0]//2
    try:
        print(de_ri[c-1:c+2,c-1:c+2])
        print(de_ti[c-1:c+2,c-1:c+2])
    except:
        # print(de_ri[c-1:c+2])
        # print(de_ti[c-1:c+2])
        print(de_ri)
        print(de_ti)

    print(a.sum(),de_ri.sum())
    print(b.sum(),de_ti.sum())
    print(a.sum()+b.sum(),de_ri.sum()+de_ti.sum())
