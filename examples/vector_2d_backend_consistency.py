import torch
torch.manual_seed(0)

import numpy as np


import meent


def run(backend):

    rcwa_options['backend'] = backend
    mee = meent.call_mee(**rcwa_options)
    mee.modeling_vector_instruction(rcwa_options, instructions)

    de_ri, de_ti = mee.conv_solve()

    return de_ri, de_ti


if __name__ == '__main__':
    rcwa_options = dict(backend=0, grating_type=2, thickness=[205, 305, 100000], period=[300, 300],
                        fourier_order=[3, 3],
                        n_I=1, n_II=1,
                        wavelength=900,
                        fft_type=2,
                        )

    si = 3.638751670074983-0.007498295841854125j
    sio2 = 1.4518-0j
    si3n4 = 2.0056-0j

    instructions = [
        # layer 1
        [sio2,
            [
                # obj 1
                ['ellipse', 75, 225, 101.5, 81.5, si, 20 * torch.pi / 180, 40, 40],
                # obj 2
                ['rectangle', 225, 75, 98.5, 81.5, si, 0, 0, 0],
            ],
        ],
        # layer 2
        [si3n4,
            [
                # obj 1
                ['rectangle', 50, 150, 31, 300, si, 0, 0, 0],
                # obj 2
                ['rectangle', 200, 150, 49.5, 300, si, 0, 0, 0],
            ],
        ],
        # layer 3
        [si,
         []
        ],
    ]

    de_ri_0, de_ti_0 = run(0)  # NumPy
    de_ri_1, de_ti_1 = run(1)  # JAX
    de_ri_2, de_ti_2 = run(2)  # PyTorch

    de_ri_1, de_ti_1 = np.array(de_ri_1), np.array(de_ti_1)
    de_ri_2, de_ti_2 = np.array(de_ri_2), np.array(de_ti_2)

    c_x, c_y = de_ri_0.shape[0] // 2, de_ri_0.shape[1] // 2

    print('Reflectance from NumPy: \n', de_ri_0[c_x-1:c_x+2, c_y-1:c_y+2])
    print('Transmittance from NumPy: \n', de_ti_0[c_x-1:c_x+2, c_y-1:c_y+2])

    print(f'Norm of difference NumPy and JAX; R: {np.linalg.norm(de_ri_0-de_ri_1)}, T: {np.linalg.norm(de_ti_0-de_ti_1)}')
    print(f'Norm of difference JAX and Torch; R: {np.linalg.norm(de_ri_1-de_ri_2)}, T: {np.linalg.norm(de_ti_1-de_ti_2)}')
    print(f'Norm of difference Torch and NumPy; R: {np.linalg.norm(de_ri_1-de_ri_2)}, T: {np.linalg.norm(de_ti_1-de_ti_2)}')

    print(0)
