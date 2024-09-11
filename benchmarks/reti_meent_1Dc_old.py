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


def test1dc_1(plot_figure=False):
    factor = 100
    option = {}
    option['pol'] = 0  # 0: TE, 1: TM
    option['n_top'] = 2.2  # n_incidence
    option['n_bot'] = 2  # n_transmission
    option['theta'] = 40 * np.pi / 180
    option['phi'] = 20 * np.pi / 180
    option['fto'] = [40, 1]
    option['period'] = [770 / factor]
    option['wavelength'] = 777 / factor
    option['thickness'] = [100 / factor, ]
    option['fourier_type'] = 1

    ucell = np.array(
        [
            [[3, 3, 3, 3, 3, 1, 1, 1, 1, 1]],
        ])

    option['ucell'] = ucell

    res_z = 11
    reti = Reticolo()
    reti_de_ri, reti_de_ti, c, d, r_field_cell = reti.run_res3(**option, grating_type=1, matlab_plot_field=0, res3_npts=res_z)
    print('reti de_ri', np.array(reti_de_ri).flatten())
    print('reti de_ti', np.array(reti_de_ti).flatten())

    # Numpy
    backend = 0
    mee = meent.call_mee(backend=backend, perturbation=1E-30, **option)
    n_de_ri, n_de_ti = mee.conv_solve()
    n_field_cell = mee.calculate_field(res_z=res_z, res_y=1, res_x=50)

    print('meent de_ri', n_de_ri[n_de_ri > 1E-5])
    print('meent de_ti', n_de_ti[n_de_ti > 1E-5])

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


def test1dc_2(plot_figure=False):
    factor = 10
    option = {}
    option['pol'] = 1  # 0: TE, 1: TM
    option['n_top'] = 1  # n_incidence
    option['n_bot'] = 2  # n_transmission
    option['theta'] = 0 * np.pi / 180
    option['phi'] = 90 * np.pi / 180
    option['fto'] = [10, 0]
    option['period'] = [3000 / factor]
    option['wavelength'] = 100 / factor
    option['thickness'] = [400 / factor, ]  # final term is for h_substrate
    option['fourier_type'] = 1

    ucell = np.array(
        [
            [[3, 3, 3, 3, 3, 1, 1, 1, 1, 1]],
        ])

    option['ucell'] = ucell

    res_z = 11
    reti = Reticolo()
    reti_de_ri, reti_de_ti, c, d, r_field_cell = reti.run_res3(**option, grating_type=1, matlab_plot_field=0, res3_npts=res_z)
    print('reti de_ri', np.array(reti_de_ri).flatten())
    print('reti de_ti', np.array(reti_de_ti).flatten())

    # Numpy
    backend = 0
    mee = meent.call_mee(backend=backend, perturbation=1E-30, **option)
    n_de_ri, n_de_ti = mee.conv_solve()
    n_field_cell = mee.calculate_field(res_z=res_z, res_y=1, res_x=50)

    print('meent de_ri', n_de_ri[n_de_ri > 1E-5])
    print('meent de_ti', n_de_ti[n_de_ti > 1E-5])

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


if __name__ == '__main__':
    test1dc_1()
    test1dc_2()

