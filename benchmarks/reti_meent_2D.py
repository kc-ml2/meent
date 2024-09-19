import os
import numpy as np
import matplotlib.pyplot as plt

import meent

try:
    from benchmarks.interface.Reticolo import Reticolo

except:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))

    from Reticolo import Reticolo

# oct2py.octave.addpath(octave.genpath('E:/funcs/software/octave_calls'))


def run_2d(option, case, plot_figure=False):
    res_z = 11
    res_y = 11
    res_x = 11
    reti = Reticolo()

    (top_refl_info_te, top_tran_info_te, top_refl_info_tm, top_tran_info_tm,
     bottom_refl_info_te, bottom_tran_info_te, bottom_refl_info_tm, bottom_tran_info_tm, reti_field_cell)\
        = reti.eng.reti_2d(case, nout=9)

    reti_field_cell = reti_field_cell[res_z:-res_z].swapaxes(1, 2)
    reti_field_cell = np.flip(reti_field_cell, 0)
    reti_field_cell = reti_field_cell.conj()

    # Numpy
    mee = meent.call_mee(backend=0, **option)
    res_numpy = mee.conv_solve()
    field_cell_numpy = mee.calculate_field(res_z=res_z, res_y=res_y, res_x=res_x)

    # JAX
    mee = meent.call_mee(backend=1, **option)  # JAX
    res_jax = mee.conv_solve()
    field_cell_jax = mee.calculate_field(res_z=res_z, res_y=res_y, res_x=res_x)

    # Torch
    mee = meent.call_mee(backend=2, **option)  # PyTorch
    res_torch = mee.conv_solve()
    field_cell_torch = mee.calculate_field(res_z=res_z, res_y=res_y, res_x=res_x).numpy()

    bds = ['Numpy', 'JAX', 'Torch']
    fields = [field_cell_numpy, field_cell_jax, field_cell_torch]

    print('Norm of (meent - reti) per backend')
    for i, res_t in enumerate([res_numpy, res_jax, res_torch]):
        reti_de_ri_te, reti_de_ti_te = np.array(top_refl_info_te.efficiency).T,  np.array(top_tran_info_te.efficiency).T
        reti_de_ri_tm, reti_de_ti_tm = np.array(top_refl_info_tm.efficiency).T, np.array(top_tran_info_tm.efficiency).T

        de_ri_te, de_ti_te = np.array(res_t.res_te_inc.de_ri).T,  np.array(res_t.res_te_inc.de_ti).T
        de_ri_tm, de_ti_tm = np.array(res_t.res_tm_inc.de_ri).T,  np.array(res_t.res_tm_inc.de_ti).T

        reti_de_ri_te = reti_de_ri_te[reti_de_ri_te > 1E-5]
        reti_de_ti_te = reti_de_ti_te[reti_de_ti_te > 1E-5]
        reti_de_ri_tm = reti_de_ri_tm[reti_de_ri_tm > 1E-5]
        reti_de_ti_tm = reti_de_ti_tm[reti_de_ti_tm > 1E-5]

        de_ri_te = de_ri_te[de_ri_te > 1E-5]
        de_ti_te = de_ti_te[de_ti_te > 1E-5]
        de_ri_tm = de_ri_tm[de_ri_tm > 1E-5]
        de_ti_tm = de_ti_tm[de_ti_tm > 1E-5]

        # reti_R_s_te = top_refl_info_te.amplitude_TE
        # reti_T_s_te = top_tran_info_te.amplitude_TE
        # reti_R_p_tm = top_refl_info_tm.amplitude_TM
        # reti_T_p_tm = top_tran_info_tm.amplitude_TM
        #
        # R_s_te = res_t.res_te_inc.R_s
        # T_s_te = res_t.res_te_inc.T_s
        # R_p_tm = res_t.res_tm_inc.R_p
        # T_p_tm = res_t.res_tm_inc.T_p

        print(bds[i])
        print('n_bot', option['n_bot'])
        print(de_ri_te)
        print(reti_de_ri_te)
        print('de_ri_te', np.linalg.norm(de_ri_te - reti_de_ri_te),)

        print('de_ri_te', np.linalg.norm(de_ri_te - reti_de_ri_te),
              'de_ti_te', np.linalg.norm(de_ti_te - reti_de_ti_te),
              'de_ri_tm', np.linalg.norm(de_ri_tm - reti_de_ri_tm),
              'de_ti_tm', np.linalg.norm(de_ti_tm - reti_de_ti_tm),
              )

        for i_field in range(reti_field_cell.shape[-1]):
            res_temp = np.linalg.norm(fields[i][i_field] - reti_field_cell[i_field])
            print(f'field, {i_field+1}th: {res_temp}')

        if plot_figure:
            title = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']

            fig, axes = plt.subplots(6, 6, figsize=(10, 5))

            for ix in range(len(title)):
                r_data = reti_field_cell[:, res_y//2, :, ix]

                im = axes[ix, 0].imshow(abs(r_data) ** 2, cmap='jet', aspect='auto')
                fig.colorbar(im, ax=axes[ix, 0], shrink=1)
                im = axes[ix, 2].imshow(r_data.real, cmap='jet', aspect='auto')
                fig.colorbar(im, ax=axes[ix, 2], shrink=1)
                im = axes[ix, 4].imshow(r_data.imag, cmap='jet', aspect='auto')
                fig.colorbar(im, ax=axes[ix, 4], shrink=1)

                n_data = fields[i][:, res_y//2, :, ix]

                im = axes[ix, 1].imshow(abs(n_data) ** 2, cmap='jet', aspect='auto')
                fig.colorbar(im, ax=axes[ix, 1], shrink=1)

                im = axes[ix, 3].imshow(n_data.real, cmap='jet', aspect='auto')
                fig.colorbar(im, ax=axes[ix, 3], shrink=1)

                im = axes[ix, 5].imshow(n_data.imag, cmap='jet', aspect='auto')
                fig.colorbar(im, ax=axes[ix, 5], shrink=1)

            ix = 0
            axes[ix, 0].title.set_text('abs**2 reti')
            axes[ix, 2].title.set_text('Re, reti')
            axes[ix, 4].title.set_text('Im, reti')
            axes[ix, 1].title.set_text('abs**2 meen')
            axes[ix, 3].title.set_text('Re, meen')
            axes[ix, 5].title.set_text('Im, meen')

            plt.show()


def case_2d_1(plot_figure=False):
    factor = 1
    option = {}
    option['pol'] = 1  # 0: TE, 1: TM
    option['n_top'] = 1  # n_incidence
    option['n_bot'] = 1  # n_transmission
    option['theta'] = 20 * np.pi / 180
    option['phi'] = 33 * np.pi / 180
    option['fto'] = [11, 11]
    option['period'] = [770 / factor, 770 / factor]
    option['wavelength'] = 777 / factor
    option['thickness'] = [100 / factor, ]  # final term is for h_substrate
    option['fourier_type'] = 1

    ucell = np.array(
        [[
            [4, 4, 6, 6, 1, 1, 1, 1, 1, 1],
            [4, 4, 6, 6, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [3, 3, 3, 3, 3, 1, 1, 1, 1, 1],
            [3, 3, 3, 3, 3, 1, 1, 1, 1, 1],
            [3, 3, 3, 3, 3, 1, 1, 1, 1, 1],
            [3, 3, 3, 3, 3, 1, 1, 1, 1, 1],
            [3, 3, 3, 3, 3, 1, 1, 1, 1, 1],
        ]])

    option['ucell'] = ucell

    run_2d(option, 1, plot_figure)


def case_2d_2(plot_figure=False):
    factor = 1
    option = {}
    option['pol'] = 1  # 0: TE, 1: TM
    option['n_top'] = 1  # n_incidence
    option['n_bot'] = 1  # n_transmission
    option['theta'] = 20 * np.pi / 180
    option['phi'] = 33 * np.pi / 180
    option['fto'] = [11, 11]
    option['period'] = [770 / factor, 770 / factor]
    option['wavelength'] = 777 / factor
    option['thickness'] = [100 / factor, ]  # final term is for h_substrate
    option['fourier_type'] = 1

    ucell = np.array(
        [[
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [3, 3, 3, 3, 3, 1, 1, 1, 1, 1],
            [3, 3, 3, 3, 3, 1, 1, 1, 1, 1],
            [3, 3, 3, 3, 3, 1, 1, 1, 1, 1],
            [3, 3, 3, 3, 3, 1, 1, 1, 1, 1],
            [3, 3, 3, 3, 3, 1, 1, 1, 1, 1],
        ]])

    option['ucell'] = ucell

    run_2d(option, 2, plot_figure)


def case_2d_3(plot_figure=False):
    factor = 1
    option = {}
    option['pol'] = 1  # 0: TE, 1: TM
    option['n_top'] = 1  # n_incidence
    option['n_bot'] = 1  # n_transmission
    option['theta'] = 20 * np.pi / 180
    option['phi'] = 33 * np.pi / 180
    option['fto'] = [11, 11]
    option['period'] = [770 / factor, 770 / factor]
    option['wavelength'] = 777 / factor
    option['thickness'] = [100 / factor, ]  # final term is for h_substrate
    option['fourier_type'] = 1

    ucell = np.array(
        [[
            [4, 4, 4, 4, 4, 1, 1, 1, 1, 1],
            [4, 4, 4, 4, 4, 1, 1, 1, 1, 1],
            [4, 4, 4, 4, 4, 1, 1, 1, 1, 1],
            [4, 4, 4, 4, 4, 1, 1, 1, 1, 1],
            [4, 4, 4, 4, 4, 1, 1, 1, 1, 1],
            [4, 4, 4, 4, 4, 1, 1, 1, 1, 1],
            [4, 4, 4, 4, 4, 1, 1, 1, 1, 1],
            [4, 4, 4, 4, 4, 1, 1, 1, 1, 1],
            [4, 4, 4, 4, 4, 1, 1, 1, 1, 1],
            [4, 4, 4, 4, 4, 1, 1, 1, 1, 1],
        ]])

    option['ucell'] = ucell
    run_2d(option, 3, plot_figure)


def case_2d_4(plot_figure=False):
    factor = 1
    option = {}
    option['pol'] = 0  # 0: TE, 1: TM
    option['n_top'] = 1  # n_incidence
    option['n_bot'] = 1  # n_transmission
    option['theta'] = 0 * np.pi / 180
    option['phi'] = 0 * np.pi / 180
    option['fto'] = [11, 11]
    option['period'] = [480 / factor, 480 / factor]
    option['wavelength'] = 550 / factor
    option['thickness'] = [220 / factor, ]  # final term is for h_substrate
    option['fourier_type'] = 1

    ucell = np.array(
        [
            [[0, 0, 0, 0, 0, 0, ],
             [0, 0, 1, 1, 0, 0],
             [0, 1, 0, 0, 1, 0],
             [0, 1, 0, 0, 1, 0],
             [0, 0, 1, 1, 0, 0],
             [0, 0, 0, 0, 0, 0]]]
    ) * 3 + 1

    option['ucell'] = ucell
    run_2d(option, 4, plot_figure)


def case_2d_5(plot_figure=False):
    factor = 1
    option = {}
    option['pol'] = 0  # 0: TE, 1: TM
    option['n_top'] = 1  # n_incidence
    option['n_bot'] = 1  # n_transmission
    option['theta'] = 0 * np.pi / 180
    option['phi'] = 0 * np.pi / 180
    option['fto'] = [11, 11]
    option['period'] = [480 / factor, 480 / factor]
    option['wavelength'] = 550 / factor
    option['thickness'] = [220 / factor, ]  # final term is for h_substrate
    option['fourier_type'] = 1

    ucell = np.array(
        [
            [[0, 0, 0, 0, 0, 0, ],
             [0, 0, 1, 1, 0, 0],
             [0, 1, 0, 0, 1, 0],
             [0, 1, 0, 0, 1, 0],
             [0, 0, 1, 1, 0, 0],
             [0, 0, 0, 0, 0, 0]]]
    ) * 3 + 1

    option['ucell'] = ucell
    run_2d(option, 5, plot_figure)


def case_2d_6(plot_figure=False):
    factor = 1
    option = {}
    option['pol'] = 0  # 0: TE, 1: TM
    option['n_top'] = 1  # n_incidence
    option['n_bot'] = 1  # n_transmission
    option['theta'] = 10 * np.pi / 180
    option['phi'] = 20 * np.pi / 180
    option['fto'] = [11, 11]
    option['period'] = [480 / factor, 480 / factor]
    option['wavelength'] = 550 / factor
    option['thickness'] = [220 / factor, ]  # final term is for h_substrate
    option['fourier_type'] = 1

    ucell = [
        # layer 1
        [1,
         [
             # obj 1
             ['rectangle', 0+240, 120+240, 160, 80, 4, 0, 0, 0],
             # obj 2
             ['rectangle', 0+240, -120+240, 160, 80, 4, 0, 0, 0],
             # obj 3
             ['rectangle', 120+240, 0+240, 80, 160, 4, 0, 0, 0],
             # obj 4
             ['rectangle', -120+240, 0+240, 80, 160, 4, 0, 0, 0],
         ],
         ],
    ]

    option['ucell'] = ucell
    run_2d(option, 6, plot_figure)


def case_2d_7(plot_figure=False):
    factor = 1
    option = {}
    option['pol'] = 0  # 0: TE, 1: TM
    option['n_top'] = 1  # n_incidence
    option['n_bot'] = 1  # n_transmission
    option['theta'] = 10 * np.pi / 180
    option['phi'] = 20 * np.pi / 180
    option['fto'] = [2, 2]
    option['period'] = [480 / factor, 480 / factor]
    option['wavelength'] = 550 / factor
    option['thickness'] = [220 / factor, ]  # final term is for h_substrate
    option['fourier_type'] = 1

    ucell = [
        # layer 1
        [1,
         [
             # obj 1
             ['rectangle', 0+240, 120+240, 160, 80, 4+1j, 0, 0, 0],
             # obj 2
             ['rectangle', 0+240, -120+240, 160, 80, 4, 0, 0, 0],
             # obj 3
             ['rectangle', 120+240, 0+240, 80, 160, 4-5j, 0, 0, 0],
             # obj 4
             ['rectangle', -120+240, 0+240, 80, 160, 4, 0, 0, 0],
         ],
         ],
    ]

    option['ucell'] = ucell
    run_2d(option, 7, plot_figure)


if __name__ == '__main__':

    case_2d_1()
    case_2d_2()
    case_2d_3()
    case_2d_4()
    case_2d_5()
    case_2d_6()
    case_2d_7()
