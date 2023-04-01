import matplotlib.pyplot as plt
import numpy as np

from meent import call_mee
try:
    from benchmarks.interface.Reticolo import Reticolo
except:
    pass


def consistency(backend):
    condition = {}
    condition['grating_type'] = 0  # 0 : just 1D grating, 1 : 1D rotating grating, 2 : 2D grating
    condition['pol'] = 0  # 0: TE, 1: TM
    condition['n_I'] = 1  # n_incidence
    condition['n_II'] = 1.5  # n_transmission
    condition['theta'] = 0 * np.pi / 180
    condition['phi'] = 0 * np.pi / 180
    condition['psi'] = 0 if condition['pol'] else 90 * np.pi / 180
    condition['fourier_order'] = 40
    condition['period'] = [1000]
    condition['wavelength'] = 650
    condition['thickness'] = [500, 200, 100, 60, 432, 500]  # final term is for h_substrate

    n_1 = 1
    n_2 = 3

    ucell = np.array(
        [
            [[1, 1, 1, 1, 1, 0, 0, 1, 1, 1,]],
            [[1, 0, 0, 1, 0, 0, 0, 1, 1, 1,]],
            [[1, 1, 0, 1, 1, 1, 1, 1, 0, 1,]],
            [[1, 1, 1, 0, 1, 0, 0, 1, 1, 1,]],
            [[0, 0, 1, 0, 1, 0, 0, 1, 1, 1,]],
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]],
        ]) * (n_2 - n_1) + n_1

    condition['ucell'] = ucell

    mee = call_mee(backend=backend, **condition)

    # Reticolo
    reti = Reticolo()
    top_refl_info, top_tran_info, bottom_refl_info, bottom_tran_info, field_cell_reti = reti.run_res3(**condition)
    center = top_tran_info.shape[0] // 2
    plot_length = min(center, 2)

    # print('reti de_ri', top_refl_info)
    print('reticolo  ; de_ri.sum(), de_ti.sum():', top_refl_info.sum(), top_tran_info.sum())
    plt.plot(top_tran_info[center - plot_length:center + plot_length + 1], label='reticolo', marker=4)

    # Meent with CFT
    mee.fft_type = 1
    de_ri, de_ti = mee.conv_solve()
    # print('meent_cont de_ri', de_ri)
    print('meent_cont; de_ri.sum(), de_ti.sum():', de_ri.sum(), de_ti.sum())
    plt.plot(de_ti[center - plot_length:center + plot_length + 1], label='continuous', marker=6)

    # Meent with DFT
    mee.fft_type = 0
    de_ri, de_ti = mee.conv_solve()
    # print('meent_disc de_ri', de_ri)
    print('meent_disc; de_ri.sum(), de_ti.sum():', de_ri.sum(), de_ti.sum())
    center = de_ri.shape[0] // 2
    plt.plot(de_ti[center - plot_length:center + plot_length + 1], label='discrete', marker=5)

    plt.legend()
    plt.show()

    field_cell_meent = mee.calculate_field()
    mee.field_plot(field_cell_meent)


consistency(0)
