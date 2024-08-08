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


def test1d_1(plot_figure=False):

    factor = 1000
    option = {}
    option['pol'] = 0  # 0: TE, 1: TM
    option['n_top'] = 2  # n_incidence
    option['n_bot'] = 1  # n_transmission
    option['theta'] = 12 * np.pi / 180
    option['phi'] = 0 * np.pi / 180
    option['fto'] = 1
    option['period'] = [770/factor]
    option['wavelength'] = 777/factor
    option['thickness'] = [100/factor,]
    option['fourier_type'] = 1

    ucell = np.array(
        [
            [[3, 3, 3, 3, 3, 1, 1, 1, 1, 1]],
        ])

    option['ucell'] = ucell

    res_z = 11
    reti = Reticolo()
    reti_de_ri, reti_de_ti, c, d, r_field_cell = reti.run_res3(**option, grating_type=0, matlab_plot_field=0, res3_npts=res_z)
    print('reti de_ri', np.array(reti_de_ri).flatten())
    print('reti de_ti', np.array(reti_de_ti).flatten())

    # Numpy
    backend = 0
    nmee = meent.call_mee(backend=backend, perturbation=1E-30, **option)
    n_de_ri, n_de_ti = nmee.conv_solve()
    n_field_cell = nmee.calculate_field(res_z=res_z, res_x=50)

    print('nmeent de_ri', n_de_ri[n_de_ri > 1E-5])
    print('nmeent de_ti', n_de_ti[n_de_ti > 1E-5])

    # r_field_cell = np.moveaxis(r_field_cell, 2, 1)
    r_field_cell = r_field_cell[:, None, :, :]
    r_field_cell = r_field_cell[res_z:-res_z]
    r_field_cell = np.flip(r_field_cell, 0)
    r_field_cell = r_field_cell.conj()

    for i in range(r_field_cell.shape[-1]):
        print(i, np.linalg.norm(r_field_cell[:, :, :, i] - n_field_cell[:, :, :, i]))

    if plot_figure:

        if option['pol'] == 0:  # TE
            title = ['1D Ey', '1D Hx', '1D Hz', ]
        else:  # TM
            title = ['1D Hy', '1D Ex', '1D Ez', ]

        fig, axes = plt.subplots(3, 6, figsize=(10, 5))

        for ix in range(len(title)):
            r_data = r_field_cell[:, 0, :, ix]

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

        axes[0, 0].title.set_text('abs**2 reti')
        axes[0, 2].title.set_text('Re, reti')
        axes[0, 4].title.set_text('Im, reti')
        axes[0, 1].title.set_text('abs**2 meen')
        axes[0, 3].title.set_text('Re, meen')
        axes[0, 5].title.set_text('Im, meen')

        plt.show()


def test1d_2(plot_figure=False):

    factor = 1
    option = {}
    option['pol'] = 1  # 0: TE, 1: TM
    option['n_top'] = 1  # n_incidence
    option['n_bot'] = 2.2  # n_transmission
    option['theta'] = 0 * np.pi / 180
    option['phi'] = 0 * np.pi / 180
    option['fto'] = 80
    option['period'] = [770/factor]
    option['wavelength'] = 777/factor
    option['thickness'] = [100/factor,]
    option['fourier_type'] = 1

    ucell = np.array(
        [
            [[3, 3, 3, 3, 3, 1, 1, 1, 1, 1]],
        ])

    option['ucell'] = ucell

    res_z = 11
    reti = Reticolo()
    reti_de_ri, reti_de_ti, c, d, r_field_cell = reti.run_res3(**option, grating_type=0, matlab_plot_field=0, res3_npts=res_z)
    print('reti de_ri', np.array(reti_de_ri).flatten())
    print('reti de_ti', np.array(reti_de_ti).flatten())

    # Numpy
    backend = 0
    nmee = meent.call_mee(backend=backend, perturbation=1E-30, **option)
    n_de_ri, n_de_ti = nmee.conv_solve()
    n_field_cell = nmee.calculate_field(res_z=res_z, res_x=50)

    print('nmeent de_ri', n_de_ri[n_de_ri > 1E-5])
    print('nmeent de_ti', n_de_ti[n_de_ti > 1E-5])

    # r_field_cell = np.moveaxis(r_field_cell, 2, 1)
    r_field_cell = r_field_cell[:, None, :, :]
    r_field_cell = r_field_cell[res_z:-res_z]
    r_field_cell = np.flip(r_field_cell, 0)
    r_field_cell = r_field_cell.conj()

    for i in range(r_field_cell.shape[-1]):
        print(i, np.linalg.norm(r_field_cell[:, :, :, i] - n_field_cell[:, :, :, i]))

    if plot_figure:

        if option['pol'] == 0:  # TE
            title = ['1D Ey', '1D Hx', '1D Hz', ]
        else:  # TM
            title = ['1D Hy', '1D Ex', '1D Ez', ]

        fig, axes = plt.subplots(3, 6, figsize=(10, 5))

        for ix in range(len(title)):
            r_data = r_field_cell[:, 0, :, ix]

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

        axes[0, 0].title.set_text('abs**2 reti')
        axes[0, 2].title.set_text('Re, reti')
        axes[0, 4].title.set_text('Im, reti')
        axes[0, 1].title.set_text('abs**2 meen')
        axes[0, 3].title.set_text('Re, meen')
        axes[0, 5].title.set_text('Im, meen')

        plt.show()


if __name__ == '__main__':
    test1d_1(False)
    test1d_2(False)
