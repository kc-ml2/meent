import torch
import numpy as np

import meent


def run_vector(rcwa_options, backend):

    rcwa_options['backend'] = backend
    mee = meent.call_mee(**rcwa_options)
    mee.modeling_vector_instruction(rcwa_options, instructions)

    de_ri, de_ti = mee.conv_solve()

    return de_ri, de_ti


def run_raster(rcwa_options, backend, fft_type):

    # ucell = ucell.numpy()

    rcwa_options['backend'] = backend
    rcwa_options['fourier_type'] = fft_type
    # 0: Discrete Fourier series; 1 is for Continuous FS which is used in vector modeling.


    if backend == 0:
        ucell = np.asarray(rcwa_options['ucell'])
    elif backend == 1:
        ucell = np.asarray(rcwa_options['ucell'])
    elif backend == 2:
        ucell = torch.as_tensor(rcwa_options['ucell'])
    else:
        raise ValueError

    rcwa_options['ucell'] = ucell

    mee = meent.call_mee(**rcwa_options)

    de_ri, de_ti = mee.conv_solve()
    return de_ri, de_ti


if __name__ == '__main__':
    rcwa_options = dict(backend=0, grating_type=2, thickness=[205, 100000], period=[300, 300],
                        fourier_order=[3, 0],
                        n_I=1, n_II=1,
                        wavelength=900,
                        fft_type=2,
                        )

    si = 3.638751670074983-0.007498295841854125j
    sio2 = 1.4518
    si3n4 = 2.0056

    instructions = [
        # layer 1
        [si3n4,
            [
                # obj 1
                ['rectangle', 50, 150, 100, 300, si, 0, 0, 0],
                # obj 2
                ['rectangle', 200, 150, 40, 300, si, 0, 0, 0],
            ],
        ],
        # layer 2
        [si,
         []
        ],
    ]

    a = si3n4
    b = si
    c = si

    ucell = [
        [
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
            [c,c,c,c,c,c,c,c,c,c] + [a,a,a,a,a,a,a,a,c,c] + [c,c,a,a,a,a,a,a,a,a],
        ],
        [
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
            [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b] + [b,b,b,b,b,b,b,b,b,b],
        ],
    ]
    # ucell = np.array(ucell)

    de_ri_v_0, de_ti_v_0 = run_vector(rcwa_options, 0)  # NumPy
    de_ri_v_1, de_ti_v_1 = run_vector(rcwa_options, 1)  # JAX
    de_ri_v_2, de_ti_v_2 = run_vector(rcwa_options, 2)  # PyTorch

    de_ri_v_1, de_ti_v_1 = np.array(de_ri_v_1), np.array(de_ti_v_1)
    de_ri_v_2, de_ti_v_2 = np.array(de_ri_v_2), np.array(de_ti_v_2)

    c_x, c_y = de_ri_v_0.shape[0] // 2, de_ri_v_0.shape[1] // 2

    print('\nVector; R and T from NumPy\n')
    print('Reflectance from NumPy: \n', de_ri_v_0[c_x-1:c_x+2, c_y-1:c_y+2])
    print('Transmittance from NumPy: \n', de_ti_v_0[c_x-1:c_x+2, c_y-1:c_y+2])

    print('\nvector across backends\n')
    print(f'Norm of difference NumPy and JAX; R: {np.linalg.norm(de_ri_v_0-de_ri_v_1)}, T: {np.linalg.norm(de_ti_v_0-de_ti_v_1)}')
    print(f'Norm of difference JAX and Torch; R: {np.linalg.norm(de_ri_v_1-de_ri_v_2)}, T: {np.linalg.norm(de_ti_v_1-de_ti_v_2)}')
    print(f'Norm of difference Torch and NumPy; R: {np.linalg.norm(de_ri_v_1-de_ri_v_2)}, T: {np.linalg.norm(de_ti_v_1-de_ti_v_2)}')

    rcwa_options['ucell'] = ucell
    de_ri_r_0_dfs, de_ti_r_0_dfs = run_raster(rcwa_options, 0, 0)  # NumPy
    de_ri_r_0_cfs, de_ti_r_0_cfs = run_raster(rcwa_options, 0, 1)  # NumPy
    de_ri_r_1_dfs, de_ti_r_1_dfs = run_raster(rcwa_options, 1, 0)  # JAX
    de_ri_r_1_cfs, de_ti_r_1_cfs = run_raster(rcwa_options, 1, 1)  # JAX
    de_ri_r_2_dfs, de_ti_r_2_dfs = run_raster(rcwa_options, 2, 0)  # PyTorch
    de_ri_r_2_cfs, de_ti_r_2_cfs = run_raster(rcwa_options, 2, 1)  # PyTorch

    de_ri_r_1_dfs, de_ti_r_1_dfs = np.array(de_ri_r_1_dfs), np.array(de_ti_r_1_dfs)
    de_ri_r_1_cfs, de_ti_r_1_cfs = np.array(de_ri_r_1_cfs), np.array(de_ti_r_1_cfs)
    de_ri_r_2_dfs, de_ti_r_2_dfs = np.array(de_ri_r_2_dfs), np.array(de_ti_r_2_dfs)
    de_ri_r_2_cfs, de_ti_r_2_cfs = np.array(de_ri_r_2_cfs), np.array(de_ti_r_2_cfs)

    print('\nRaster with DFS; R and T from NumPy\n')
    print('Reflectance from NumPy: \n', de_ri_r_1_dfs[c_x-1:c_x+2, c_y-1:c_y+2])
    print('Transmittance from NumPy: \n', de_ti_r_1_dfs[c_x-1:c_x+2, c_y-1:c_y+2])

    print('\nRaster with CFS; R and T from NumPy\n')
    print('Reflectance from NumPy: \n', de_ri_r_1_dfs[c_x-1:c_x+2, c_y-1:c_y+2])
    print('Transmittance from NumPy: \n', de_ti_r_1_dfs[c_x-1:c_x+2, c_y-1:c_y+2])

    print('\nvector vs raster dfs \n')
    print(f'Norm of difference, vector and raster dfs, NumPy;'
          f' R: {np.linalg.norm(de_ri_v_0-de_ri_r_0_dfs)}, T: {np.linalg.norm(de_ti_v_0-de_ti_r_0_dfs)}')
    print(f'Norm of difference, vector and raster dfs, JAX;'
          f' R: {np.linalg.norm(de_ri_v_1-de_ri_r_1_dfs)}, T: {np.linalg.norm(de_ti_v_1-de_ti_r_1_dfs)}')
    print(f'Norm of difference, vector and raster dfs, Torch;'
          f' R: {np.linalg.norm(de_ri_v_2-de_ri_r_2_dfs)}, T: {np.linalg.norm(de_ti_v_2-de_ti_r_2_dfs)}')

    print('\nvector vs raster cfs \n')
    print(f'Norm of difference, vector and raster cfs, NumPy;'
          f' R: {np.linalg.norm(de_ri_v_0-de_ri_r_0_cfs)}, T: {np.linalg.norm(de_ti_v_0-de_ti_r_0_cfs)}')
    print(f'Norm of difference, vector and raster cfs, JAX;'
          f' R: {np.linalg.norm(de_ri_v_1-de_ri_r_1_cfs)}, T: {np.linalg.norm(de_ti_v_1-de_ti_r_1_cfs)}')
    print(f'Norm of difference, vector and raster cfs, Torch;'
          f' R: {np.linalg.norm(de_ri_v_2-de_ri_r_2_cfs)}, T: {np.linalg.norm(de_ti_v_2-de_ti_r_2_cfs)}')

    print(0)
