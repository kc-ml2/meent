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


def test2d_1(plot_figure=False):
    reti = Reticolo()

    [top_refl_info, top_tran_info, bottom_refl_info, bottom_tran_info, field_cell] = reti.eng.reti_2d(1, nout=5)
    reti_de_ri, reti_de_ti, c, d, r_field_cell = top_refl_info.efficiency, top_tran_info.efficiency, bottom_refl_info.efficiency, \
        bottom_tran_info.efficiency, field_cell

    factor = 1
    option = {}
    option['grating_type'] = 2  # 0 : just 1D grating, 1 : 1D rotating grating, 2 : 2D grating
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

    print('reti de_ri', np.array(reti_de_ri))
    print('reti de_ti', np.array(reti_de_ti))

    res_z = 11

    # Numpy
    backend = 0
    nmee = meent.call_mee(backend=backend, **option)
    n_de_ri, n_de_ti = nmee.conv_solve()
    n_field_cell = nmee.calculate_field(res_z=res_z, res_y=50, res_x=50)
    # print('nmeent de_ri', n_de_ri)
    # print('nmeent de_ti', n_de_ti)
    print('nmeent de_ri', n_de_ri[n_de_ri > 1E-5])
    print('nmeent de_ti', n_de_ti[n_de_ti > 1E-5])

    r_field_cell = np.moveaxis(r_field_cell, 2, 1)
    r_field_cell = r_field_cell[res_z:-res_z]
    r_field_cell = np.flip(r_field_cell, 0)
    r_field_cell = r_field_cell.conj()

    title = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']

    for i in range(6):
        print(i, np.linalg.norm(r_field_cell[:, :, :, i] - n_field_cell[:, :, :, i]))


    if plot_figure:
        title = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']
        fig, axes = plt.subplots(6, 6, figsize=(10, 5))

        for ix in range(len(title)):
            # r_data = np.flipud(r_field_cell[res3_npts:-res3_npts, :, 0, ix]).conj()
            r_data = r_field_cell[:, 0, :, ix]
            im = axes[ix, 0].imshow(abs(r_data) ** 2, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 0], shrink=1)
            im = axes[ix, 2].imshow(r_data.real, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 2], shrink=1)
            im = axes[ix, 4].imshow(r_data.imag, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 4], shrink=1)

            n_data = n_field_cell[:, 0, :, ix]

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

        fig, axes = plt.subplots(6, 6, figsize=(10, 5))

        for ix in range(len(title)):
            # r_data = np.transpose(r_field_cell[2*res3_npts, :, :, ix]).conj()
            r_data = r_field_cell[5, :, :, ix]

            im = axes[ix, 0].imshow(abs(r_data) ** 2, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 0], shrink=1)
            im = axes[ix, 2].imshow(r_data.real, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 2], shrink=1)
            im = axes[ix, 4].imshow(r_data.imag, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 4], shrink=1)

            n_data = n_field_cell[5, :, :, ix]

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

    return


def test2d_2(plot_figure=False):
    reti = Reticolo()

    [top_refl_info, top_tran_info, bottom_refl_info, bottom_tran_info, field_cell] = reti.eng.reti_2d(2, nout=5)
    reti_de_ri, reti_de_ti, c, d, r_field_cell = top_refl_info.efficiency, top_tran_info.efficiency, bottom_refl_info.efficiency, \
        bottom_tran_info.efficiency, field_cell

    factor = 1
    option = {}
    option['grating_type'] = 2  # 0 : just 1D grating, 1 : 1D rotating grating, 2 : 2D grating
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
        [
            [
                [0, 1, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 1, 0],
            ]]
    ) * (3) + 1

    ucell = np.array(
        [
            [[0, 0, 0, 0, 0, 0, ],
             [0, 0, 1, 1, 0, 0],
             [0, 1, 0, 0, 1, 0],
             [0, 1, 0, 0, 1, 0],
             [0, 0, 1, 1, 0, 0],
             [0, 0, 0, 0, 0, 0]]]
    ) * (3) + 1
    ucell = np.array(
        [[
            [3, 3, 3, 3, 3, 1, 1, 1, 1, 1],
            [3, 3, 3, 3, 3, 1, 1, 1, 1, 1],
            # [3, 3, 3, 3, 3, 1, 1, 1, 1, 1],
            # [3, 3, 3, 3, 3, 1, 1, 1, 1, 1],
            # [3, 3, 3, 3, 3, 1, 1, 1, 1, 1],
            # [3, 3, 3, 3, 3, 1, 1, 1, 1, 1],
            # [3, 3, 3, 3, 3, 1, 1, 1, 1, 1],
            # [3, 3, 3, 3, 3, 1, 1, 1, 1, 1],
            # [3, 3, 3, 3, 3, 1, 1, 1, 1, 1],
            # [3, 3, 3, 3, 3, 1, 1, 1, 1, 1],
        ]])

    # ucell = np.array(
    #     [[
    #         [3, 3, 3, 3, 3, 1, 1, 1, 1, 1],
    #         [3, 3, 3, 3, 3, 1, 1, 1, 1, 1],
    #         [3, 3, 3, 3, 3, 1, 1, 1, 1, 1],
    #         [3, 3, 3, 3, 3, 1, 1, 1, 1, 1],
    #         [3, 3, 3, 3, 3, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     ]])


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

    # ucell = np.array(
    #     [[
    #         [4, 4, 6, 6, 1, 1, 1, 1, 1, 1],
    #         [4, 4, 6, 6, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [3, 3, 3, 3, 3, 1, 1, 1, 1, 1],
    #         [3, 3, 3, 3, 3, 1, 1, 1, 1, 1],
    #         [3, 3, 3, 3, 3, 1, 1, 1, 1, 1],
    #         [3, 3, 3, 3, 3, 1, 1, 1, 1, 1],
    #         [3, 3, 3, 3, 3, 1, 1, 1, 1, 1],
    #     ]])

    # ucell = np.array(
    #     [[
    #         [1, 1, 3, 1, 1, 1, 1, 1, 3, 1],
    #         [4, 1, 3, 1, 1, 1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [8, 1, 1, 1, 1, 1, 1, 1, 5, 1],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #         [3, 3, 3, 3, 3, 1, 1, 1, 1, 1],
    #         [3, 3, 3, 3, 3, 1, 1, 1, 1, 1],
    #         [3, 3, 3, 3, 3, 1, 1, 1, 1, 1],
    #         [3, 3, 3, 3, 3, 1, 1, 1, 1, 1],
    #         [3, 3, 3, 3, 3, 1, 1, 1, 1, 1],
    #     ]])


    # ucell = np.repeat(ucell, 10, axis=2)
    option['ucell'] = ucell

    # reti = Reticolo()
    # reti_de_ri, reti_de_ti, c, d, r_field_cell = reti.run_res3(**option, res3_npts=res3_npts)
    print('reti de_ri', np.array(reti_de_ri))
    print('reti de_ti', np.array(reti_de_ti))
    # print('reti de_ri', np.array(reti_de_ri).flatten())
    # print('reti de_ti', np.array(reti_de_ti).flatten())

    res_z = 11

    # Numpy
    backend = 0
    nmee = meent.call_mee(backend=backend, **option)
    n_de_ri, n_de_ti = nmee.conv_solve()
    n_field_cell = nmee.calculate_field(res_z=res_z, res_y=50, res_x=50)
    # print('nmeent de_ri', n_de_ri)
    # print('nmeent de_ti', n_de_ti)
    print('nmeent de_ri', n_de_ri[n_de_ri > 1E-5])
    print('nmeent de_ti', n_de_ti[n_de_ti > 1E-5])

    r_field_cell = np.moveaxis(r_field_cell, 2, 1)
    r_field_cell = r_field_cell[res_z:-res_z]
    r_field_cell = np.flip(r_field_cell, 0)
    r_field_cell = r_field_cell.conj()

    title = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']

    for i in range(6):
        print(i, np.linalg.norm(r_field_cell[:, :, :, i] - n_field_cell[:, :, :, i]))

    if plot_figure:
        title = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']
        fig, axes = plt.subplots(6, 6, figsize=(10, 5))

        for ix in range(len(title)):
            # r_data = np.flipud(r_field_cell[res3_npts:-res3_npts, :, 0, ix]).conj()
            r_data = r_field_cell[:, 0, :, ix]
            im = axes[ix, 0].imshow(abs(r_data) ** 2, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 0], shrink=1)
            im = axes[ix, 2].imshow(r_data.real, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 2], shrink=1)
            im = axes[ix, 4].imshow(r_data.imag, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 4], shrink=1)

            n_data = n_field_cell[:, 0, :, ix]

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

        fig, axes = plt.subplots(6, 6, figsize=(10, 5))

        for ix in range(len(title)):
            # r_data = np.transpose(r_field_cell[2*res3_npts, :, :, ix]).conj()
            r_data = r_field_cell[5, :, :, ix]

            im = axes[ix, 0].imshow(abs(r_data) ** 2, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 0], shrink=1)
            im = axes[ix, 2].imshow(r_data.real, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 2], shrink=1)
            im = axes[ix, 4].imshow(r_data.imag, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 4], shrink=1)

            n_data = n_field_cell[5, :, :, ix]

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

    return


def test2d_3(plot_figure=False):
    reti = Reticolo()

    [top_refl_info, top_tran_info, bottom_refl_info, bottom_tran_info, field_cell] = reti.eng.reti_2d(3, nout=5)
    reti_de_ri, reti_de_ti, c, d, r_field_cell = top_refl_info.efficiency, top_tran_info.efficiency, bottom_refl_info.efficiency, \
        bottom_tran_info.efficiency, field_cell

    factor = 1
    option = {}
    option['grating_type'] = 2  # 0 : just 1D grating, 1 : 1D rotating grating, 2 : 2D grating
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
        [
            [
                [0, 1, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 1, 0],
            ]]
    ) * (3) + 1

    ucell = np.array(
        [
            [[0, 0, 0, 0, 0, 0, ],
             [0, 0, 1, 1, 0, 0],
             [0, 1, 0, 0, 1, 0],
             [0, 1, 0, 0, 1, 0],
             [0, 0, 1, 1, 0, 0],
             [0, 0, 0, 0, 0, 0]]]
    ) * (3) + 1

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

    # reti = Reticolo()
    # reti_de_ri, reti_de_ti, c, d, r_field_cell = reti.run_res3(**option, res3_npts=res3_npts)
    print('reti de_ri', np.array(reti_de_ri))
    print('reti de_ti', np.array(reti_de_ti))
    # print('reti de_ri', np.array(reti_de_ri).flatten())
    # print('reti de_ti', np.array(reti_de_ti).flatten())

    res_z = 11

    # Numpy
    backend = 0
    nmee = meent.call_mee(backend=backend, **option)
    n_de_ri, n_de_ti = nmee.conv_solve()
    n_field_cell = nmee.calculate_field(res_z=res_z, res_y=50, res_x=50)
    # print('nmeent de_ri', n_de_ri)
    # print('nmeent de_ti', n_de_ti)
    print('nmeent de_ri', n_de_ri[n_de_ri > 1E-5])
    print('nmeent de_ti', n_de_ti[n_de_ti > 1E-5])

    r_field_cell = np.moveaxis(r_field_cell, 2, 1)
    r_field_cell = r_field_cell[res_z:-res_z]
    r_field_cell = np.flip(r_field_cell, 0)
    r_field_cell = r_field_cell.conj()

    title = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']

    for i in range(6):
        print(i, np.linalg.norm(r_field_cell[:, :, :, i] - n_field_cell[:, :, :, i]))

    if plot_figure:
        title = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']
        fig, axes = plt.subplots(6, 6, figsize=(10, 5))

        for ix in range(len(title)):
            # r_data = np.flipud(r_field_cell[res3_npts:-res3_npts, :, 0, ix]).conj()
            r_data = r_field_cell[:, 0, :, ix]
            im = axes[ix, 0].imshow(abs(r_data) ** 2, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 0], shrink=1)
            im = axes[ix, 2].imshow(r_data.real, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 2], shrink=1)
            im = axes[ix, 4].imshow(r_data.imag, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 4], shrink=1)

            n_data = n_field_cell[:, 0, :, ix]

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

        fig, axes = plt.subplots(6, 6, figsize=(10, 5))

        for ix in range(len(title)):
            # r_data = np.transpose(r_field_cell[2*res3_npts, :, :, ix]).conj()
            r_data = r_field_cell[5, :, :, ix]

            im = axes[ix, 0].imshow(abs(r_data) ** 2, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 0], shrink=1)
            im = axes[ix, 2].imshow(r_data.real, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 2], shrink=1)
            im = axes[ix, 4].imshow(r_data.imag, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 4], shrink=1)

            n_data = n_field_cell[5, :, :, ix]

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

    return


def test2d_4(plot_figure=False):
    reti = Reticolo()

    [top_refl_info, top_tran_info, bottom_refl_info, bottom_tran_info, field_cell] = reti.eng.reti_2d(4, nout=5)
    reti_de_ri, reti_de_ti, c, d, r_field_cell = top_refl_info.efficiency, top_tran_info.efficiency, bottom_refl_info.efficiency, \
        bottom_tran_info.efficiency, field_cell

    factor = 1
    option = {}
    option['grating_type'] = 2  # 0 : just 1D grating, 1 : 1D rotating grating, 2 : 2D grating
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
            [
                [0, 1, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 1, 0],
            ]]
    ) * (3) + 1

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

    # reti = Reticolo()
    # reti_de_ri, reti_de_ti, c, d, r_field_cell = reti.run_res3(**option, res3_npts=res3_npts)
    print('reti de_ri', np.array(reti_de_ri))
    print('reti de_ti', np.array(reti_de_ti))
    # print('reti de_ri', np.array(reti_de_ri).flatten())
    # print('reti de_ti', np.array(reti_de_ti).flatten())

    res_z = 11

    # Numpy
    backend = 0
    nmee = meent.call_mee(backend=backend, **option)
    n_de_ri, n_de_ti = nmee.conv_solve()
    n_field_cell = nmee.calculate_field(res_z=res_z, res_y=50, res_x=50)
    # print('nmeent de_ri', n_de_ri)
    # print('nmeent de_ti', n_de_ti)
    print('nmeent de_ri', n_de_ri[n_de_ri > 1E-5])
    print('nmeent de_ti', n_de_ti[n_de_ti > 1E-5])

    r_field_cell = np.moveaxis(r_field_cell, 2, 1)
    r_field_cell = r_field_cell[res_z:-res_z]
    r_field_cell = np.flip(r_field_cell, 0)
    r_field_cell = r_field_cell.conj()

    title = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']

    for i in range(6):
        print(i, np.linalg.norm(r_field_cell[:, :, :, i] - n_field_cell[:, :, :, i]))


    if plot_figure:
        title = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']
        fig, axes = plt.subplots(6, 6, figsize=(10, 5))

        for ix in range(len(title)):
            # r_data = np.flipud(r_field_cell[res3_npts:-res3_npts, :, 0, ix]).conj()
            r_data = r_field_cell[:, 0, :, ix]
            im = axes[ix, 0].imshow(abs(r_data) ** 2, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 0], shrink=1)
            im = axes[ix, 2].imshow(r_data.real, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 2], shrink=1)
            im = axes[ix, 4].imshow(r_data.imag, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 4], shrink=1)

            n_data = n_field_cell[:, 0, :, ix]

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

        fig, axes = plt.subplots(6, 6, figsize=(10, 5))

        for ix in range(len(title)):
            # r_data = np.transpose(r_field_cell[2*res3_npts, :, :, ix]).conj()
            r_data = r_field_cell[5, :, :, ix]

            im = axes[ix, 0].imshow(abs(r_data) ** 2, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 0], shrink=1)
            im = axes[ix, 2].imshow(r_data.real, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 2], shrink=1)
            im = axes[ix, 4].imshow(r_data.imag, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 4], shrink=1)

            n_data = n_field_cell[5, :, :, ix]

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

    return


def test2d_5(plot_figure=False):
    reti = Reticolo()

    [top_refl_info, top_tran_info, bottom_refl_info, bottom_tran_info, field_cell] = reti.eng.reti_2d(5, nout=5)
    reti_de_ri, reti_de_ti, c, d, r_field_cell = top_refl_info.efficiency, top_tran_info.efficiency, bottom_refl_info.efficiency, \
        bottom_tran_info.efficiency, field_cell

    factor = 1
    option = {}
    option['grating_type'] = 2  # 0 : just 1D grating, 1 : 1D rotating grating, 2 : 2D grating
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
            [
                [0, 1, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 1, 0],
            ]]
    ) * (3) + 1

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

    # reti = Reticolo()
    # reti_de_ri, reti_de_ti, c, d, r_field_cell = reti.run_res3(**option, res3_npts=res3_npts)
    print('reti de_ri', np.array(reti_de_ri))
    print('reti de_ti', np.array(reti_de_ti))
    # print('reti de_ri', np.array(reti_de_ri).flatten())
    # print('reti de_ti', np.array(reti_de_ti).flatten())

    res_z = 11

    # Numpy
    backend = 0
    nmee = meent.call_mee(backend=backend, **option)
    n_de_ri, n_de_ti = nmee.conv_solve()
    n_field_cell = nmee.calculate_field(res_z=res_z, res_y=50, res_x=50)
    # print('nmeent de_ri', n_de_ri)
    # print('nmeent de_ti', n_de_ti)
    print('nmeent de_ri', n_de_ri[n_de_ri > 1E-5])
    print('nmeent de_ti', n_de_ti[n_de_ti > 1E-5])

    r_field_cell = np.moveaxis(r_field_cell, 2, 1)
    r_field_cell = r_field_cell[res_z:-res_z]
    r_field_cell = np.flip(r_field_cell, 0)
    r_field_cell = r_field_cell.conj()

    for i in range(6):
        print(i, np.linalg.norm(r_field_cell[:, :, :, i] - n_field_cell[:, :, :, i]))

    if plot_figure:
        title = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']
        fig, axes = plt.subplots(6, 6, figsize=(10, 5))

        for ix in range(len(title)):
            # r_data = np.flipud(r_field_cell[res3_npts:-res3_npts, :, 0, ix]).conj()
            r_data = r_field_cell[:, 0, :, ix]
            im = axes[ix, 0].imshow(abs(r_data) ** 2, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 0], shrink=1)
            im = axes[ix, 2].imshow(r_data.real, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 2], shrink=1)
            im = axes[ix, 4].imshow(r_data.imag, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 4], shrink=1)

            n_data = n_field_cell[:, 0, :, ix]

            im = axes[ix, 1].imshow(abs(n_data) ** 2, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 1], shrink=1)

            im = axes[ix, 3].imshow(n_data.real, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 3], shrink=1)

            im = axes[ix, 5].imshow(n_data.imag, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 5], shrink=1)

        axes[0, 0].title.set_text('abs**2 reti')
        axes[0, 2].title.set_text('Re, reti')
        axes[0, 4].title.set_text('Im, reti')
        axes[0, 1].title.set_text('abs**2 meen')
        axes[0, 3].title.set_text('Re, meen')
        axes[0, 5].title.set_text('Im, meen')

        plt.show()

        fig, axes = plt.subplots(6, 6, figsize=(10, 5))

        for ix in range(len(title)):
            # r_data = np.transpose(r_field_cell[2*res3_npts, :, :, ix]).conj()
            r_data = r_field_cell[5, :, :, ix]

            im = axes[ix, 0].imshow(abs(r_data) ** 2, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 0], shrink=1)
            im = axes[ix, 2].imshow(r_data.real, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 2], shrink=1)
            im = axes[ix, 4].imshow(r_data.imag, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 4], shrink=1)

            n_data = n_field_cell[5, :, :, ix]

            im = axes[ix, 1].imshow(abs(n_data) ** 2, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 1], shrink=1)

            im = axes[ix, 3].imshow(n_data.real, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 3], shrink=1)

            im = axes[ix, 5].imshow(n_data.imag, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 5], shrink=1)

        axes[0, 0].title.set_text('abs**2 reti')
        axes[0, 2].title.set_text('Re, reti')
        axes[0, 4].title.set_text('Im, reti')
        axes[0, 1].title.set_text('abs**2 meen')
        axes[0, 3].title.set_text('Re, meen')
        axes[0, 5].title.set_text('Im, meen')

        plt.show()

    return


def test2d_6(plot_figure=False):

    res_z = 11

    reti = Reticolo()

    [top_refl_info, top_tran_info, bottom_refl_info, bottom_tran_info, field_cell] = reti.eng.reti_2d(6, nout=5)
    reti_de_ri, reti_de_ti, c, d, r_field_cell = top_refl_info.efficiency, top_tran_info.efficiency, bottom_refl_info.efficiency, \
        bottom_tran_info.efficiency, field_cell
    print('reti de_ri', np.array(reti_de_ri))
    print('reti de_ti', np.array(reti_de_ti))
    r_field_cell = np.moveaxis(r_field_cell, 2, 1)
    r_field_cell = r_field_cell[res_z:-res_z]
    r_field_cell = np.flip(r_field_cell, 0)
    r_field_cell = r_field_cell.conj()

    factor = 1
    option = {}
    option['grating_type'] = 2  # 0 : just 1D grating, 1 : 1D rotating grating, 2 : 2D grating
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

    # Numpy
    backend = 0
    mee = meent.call_mee(backend=backend, **option)

    instructions = [
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

    # instructions = [
    #     # layer 1
    #     [1,
    #      [
    #          # obj 1
    #          ['ellipse', 75, 225, 101.5, 81.5, si, 20 * np.pi / 180, 40, 40],
    #          # obj 2
    #          ['rectangle', 225, 75, 98.5, 81.5, si, 0, 0, 0],
    #      ],
    #      ],
    #     # layer 2
    #     [si3n4,
    #      [
    #          # obj 1
    #          ['rectangle', 50, 150, 31, 300, si, 0, 0, 0],
    #          # obj 2
    #          ['rectangle', 200, 150, 49.5, 300, si, 0, 0, 0],
    #      ],
    #      ],
    #     # layer 3
    #     [si,
    #      []
    #      ],
    # ]

    mee.modeling_vector_instruction(instructions)

    n_de_ri, n_de_ti = mee.conv_solve()
    n_field_cell = mee.calculate_field(res_z=res_z, res_y=50, res_x=50)
    # print('nmeent de_ri', n_de_ri)
    # print('nmeent de_ti', n_de_ti)
    print('nmeent de_ri', n_de_ri[n_de_ri > 1E-5])
    print('nmeent de_ti', n_de_ti[n_de_ti > 1E-5])

    for i in range(6):
        print(i, np.linalg.norm(r_field_cell[:, :, :, i] - n_field_cell[:, :, :, i]))

    if plot_figure:
        title = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']
        fig, axes = plt.subplots(6, 6, figsize=(10, 5))

        for ix in range(len(title)):
            r_data = r_field_cell[:, 0, :, ix]
            im = axes[ix, 0].imshow(abs(r_data) ** 2, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 0], shrink=1)
            im = axes[ix, 2].imshow(r_data.real, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 2], shrink=1)
            im = axes[ix, 4].imshow(r_data.imag, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 4], shrink=1)

            n_data = n_field_cell[:, 0, :, ix]

            im = axes[ix, 1].imshow(abs(n_data) ** 2, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 1], shrink=1)

            im = axes[ix, 3].imshow(n_data.real, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 3], shrink=1)

            im = axes[ix, 5].imshow(n_data.imag, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 5], shrink=1)

        axes[0, 0].title.set_text('abs**2 reti')
        axes[0, 2].title.set_text('Re, reti')
        axes[0, 4].title.set_text('Im, reti')
        axes[0, 1].title.set_text('abs**2 meen')
        axes[0, 3].title.set_text('Re, meen')
        axes[0, 5].title.set_text('Im, meen')

        plt.show()

        fig, axes = plt.subplots(6, 6, figsize=(10, 5))

        for ix in range(len(title)):
            r_data = r_field_cell[5, :, :, ix]

            im = axes[ix, 0].imshow(abs(r_data) ** 2, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 0], shrink=1)
            im = axes[ix, 2].imshow(r_data.real, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 2], shrink=1)
            im = axes[ix, 4].imshow(r_data.imag, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 4], shrink=1)

            n_data = n_field_cell[5, :, :, ix]

            im = axes[ix, 1].imshow(abs(n_data) ** 2, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 1], shrink=1)

            im = axes[ix, 3].imshow(n_data.real, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 3], shrink=1)

            im = axes[ix, 5].imshow(n_data.imag, cmap='jet', aspect='auto')
            fig.colorbar(im, ax=axes[ix, 5], shrink=1)

        axes[0, 0].title.set_text('abs**2 reti')
        axes[0, 2].title.set_text('Re, reti')
        axes[0, 4].title.set_text('Im, reti')
        axes[0, 1].title.set_text('abs**2 meen')
        axes[0, 3].title.set_text('Re, meen')
        axes[0, 5].title.set_text('Im, meen')

        plt.show()

    return


if __name__ == '__main__':

    test2d_1()
    test2d_2()
    test2d_3()
    test2d_4()
    test2d_5()
    test2d_6()
