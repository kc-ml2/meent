import torch
import numpy as np

import meent


def run_vector(rcwa_options_setting):

    instructions = [
        # layer 1
        [1,
            [
                # obj 1
                ['rectangle', 450, 450, 300, 900, 4.97 + 1j*4.2, 0 * torch.pi / 180, 0, 0],
            ],
        ],
    ]

    mee = meent.call_mee(**rcwa_options_setting)
    mee.fft_type = 2
    mee.modeling_vector_instruction(rcwa_options_setting, instructions)

    de_ri, de_ti = mee.conv_solve()

    return de_ri, de_ti


def run_raster(rcwa_options):

    ucell = torch.tensor([
        [
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
        ],
    ]) * (3.97 + 1j*4.2) + 1

    ucell = ucell.numpy()
    mee = meent.call_mee(**rcwa_options)
    mee.ucell = ucell

    mee.fft_type = 1  # 0: Discrete Fourier series; 1 is for Continuous FS which is used in vector modeling.

    de_ri, de_ti = mee.conv_solve()
    return de_ri, de_ti


if __name__ == '__main__':
    rcwa_options_setting = dict(backend=2, grating_type=2, thickness=[100], period=[900, 900], fourier_order=[2, 0],
                        n_I=1, n_II=1, wavelength=900)

    de_ri_vector, de_ti_vector = run_vector(rcwa_options_setting)
    de_ri_raster, de_ti_raster = run_raster(rcwa_options_setting)
    print(de_ri_vector)
    print(de_ti_vector)
    print(de_ri_raster)
    print(de_ti_raster)
    # print(torch.norm(de_ri_vector-de_ri_raster))
    # print(torch.norm(de_ti_vector-de_ti_raster))
    print(np.linalg.norm(de_ri_vector-de_ri_raster))
    print(np.linalg.norm(de_ti_vector-de_ti_raster))
    print(0)
