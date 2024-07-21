import matplotlib.pyplot as plt
import numpy as np

from meent import call_mee
from benchmarks.interface.Reticolo import Reticolo


def consistency(backend):
    option = {}
    option['grating_type'] = 0  # 0 : just 1D grating, 1 : 1D rotating grating, 2 : 2D grating
    option['pol'] = 0  # 0: TE, 1: TM
    option['n_top'] = 1  # n_incidence
    option['n_bot'] = 1.5  # n_transmission
    option['theta'] = 0 * np.pi / 180
    option['phi'] = 0 * np.pi / 180
    option['psi'] = 0 if option['pol'] else 90 * np.pi / 180
    option['fourier_order'] = 40
    option['period'] = [1000]
    option['wavelength'] = 650
    option['thickness'] = [500, 200, 100, 60, 432, 500]  # final term is for h_substrate

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

    option['ucell'] = ucell

    mee = call_mee(backend=backend, **option)

    # Reticolo
    reti = Reticolo()
    top_refl_info, top_tran_info, bottom_refl_info, bottom_tran_info, field_cell_reti = reti.run_res3(**option)
    center = top_tran_info.shape[0] // 2
    plot_length = min(center, 2)

    # print('reti de_ri', top_refl_info)
    print('reticolo  ; de_ri.sum(), de_ti.sum():', top_refl_info.sum(), top_tran_info.sum())
    plt.plot(top_tran_info[center - plot_length:center + plot_length + 1], label='reticolo', marker=4)

    # Meent with CFT
    mee.fourier_type = 1
    de_ri, de_ti = mee.conv_solve()
    # print('meent_cont de_ri', de_ri)
    print('meent_cont; de_ri.sum(), de_ti.sum():', de_ri.sum(), de_ti.sum())
    center = de_ri.shape[0] // 2
    plt.plot(de_ti[center - plot_length:center + plot_length + 1], label='continuous', marker=6)

    # Meent with DFT
    mee.fourier_type = 0
    de_ri, de_ti = mee.conv_solve()
    # print('meent_disc de_ri', de_ri)
    print('meent_disc; de_ri.sum(), de_ti.sum():', de_ri.sum(), de_ti.sum())
    center = de_ri.shape[0] // 2
    plt.plot(de_ti[center - plot_length:center + plot_length + 1], label='discrete', marker=5)

    plt.legend()
    plt.show()

    field_cell_meent = mee.calculate_field()
    mee.field_plot(field_cell_meent)


if __name__ == '__main__':
    try:
        print('NumpyMeent')
        consistency(0)
    except Exception as e:
        print('NumpyMeent has problem.')
        print(e)

    try:
        print('JaxMeent')
        consistency(1)
    except Exception as e:
        print('JaxMeent has problem. Do you have JAX?')
        print(e)

    try:
        print('TorchMeent')
        consistency(2)
    except Exception as e:
        print('TorchMeent has problem. Do you have PyTorch?')
        print(e)
