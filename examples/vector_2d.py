import torch

torch.manual_seed(0)

import meent


def run():
    rcwa_options = dict(backend=1, grating_type=2, thickness=[205, 305, 100000], period=[300, 300],
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

    mee = meent.call_mee(**rcwa_options)
    mee.modeling_vector_instruction(rcwa_options, instructions)

    de_ri, de_ti = mee.conv_solve()
    print(de_ri)

    return


if __name__ == '__main__':

    res = run()

    print(0)
