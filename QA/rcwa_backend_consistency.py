import numpy as np

import meent


def consistency_check(option):

    mee = meent.call_mee(backend=0, perturbation=1E-30, **option)  # NumPy
    de_ri_numpy, de_ti_numpy = mee.conv_solve()
    field_cell_numpy = mee.calculate_field(res_z=50, res_x=50)

    mee = meent.call_mee(backend=1, perturbation=1E-30, **option)  # JAX
    de_ri_jax, de_ti_jax = mee.conv_solve()
    field_cell_jax = mee.calculate_field(res_z=50, res_x=50)

    mee = meent.call_mee(backend=2, perturbation=1E-30, **option)  # PyTorch
    de_ri_torch, de_ti_torch = mee.conv_solve()
    field_cell_torch = mee.calculate_field(res_z=50, res_x=50)
    de_ri_torch, de_ti_torch = de_ri_torch.numpy(), de_ti_torch.numpy()
    field_cell_torch = field_cell_torch.numpy()

    digit = 20

    res1 = [(np.linalg.norm(de_ri_numpy - de_ri_jax) / de_ri_numpy.size).round(digit),
            (np.linalg.norm(de_ri_jax - de_ri_torch) / de_ri_numpy.size).round(digit),
            (np.linalg.norm(de_ri_torch - de_ri_numpy) / de_ri_numpy.size).round(digit),]
    res2 = [(np.linalg.norm(de_ti_numpy - de_ti_jax) / de_ti_numpy.size).round(digit),
            (np.linalg.norm(de_ti_jax - de_ti_torch) / de_ti_numpy.size).round(digit),
            (np.linalg.norm(de_ti_torch - de_ti_numpy) / de_ti_numpy.size).round(digit),]
    res3 = [(np.linalg.norm(field_cell_numpy - field_cell_jax) / field_cell_numpy.size).round(digit),
            (np.linalg.norm(field_cell_jax - field_cell_torch) / field_cell_numpy.size).round(digit),
            (np.linalg.norm(field_cell_torch - field_cell_numpy) / field_cell_numpy.size).round(digit),]

    print('Refle', res1)
    print('Trans', res2)
    print('Field', res3)


def consistency_check_vector(option, instructions):

    mee = meent.call_mee(backend=0, perturbation=1E-30, **option)  # NumPy
    mee.modeling_vector_instruction(instructions)

    de_ri_numpy, de_ti_numpy = mee.conv_solve()
    field_cell_numpy = mee.calculate_field(res_z=50, res_x=50)

    mee = meent.call_mee(backend=1, perturbation=1E-30, **option)  # JAX
    mee.modeling_vector_instruction(instructions)
    de_ri_jax, de_ti_jax = mee.conv_solve()
    field_cell_jax = mee.calculate_field(res_z=50, res_x=50)

    mee = meent.call_mee(backend=2, perturbation=1E-30, **option)  # PyTorch
    mee.modeling_vector_instruction(instructions)
    de_ri_torch, de_ti_torch = mee.conv_solve()
    field_cell_torch = mee.calculate_field(res_z=50, res_x=50)
    de_ri_torch, de_ti_torch = de_ri_torch.numpy(), de_ti_torch.numpy()
    field_cell_torch = field_cell_torch.numpy()

    digit = 20

    res1 = [(np.linalg.norm(de_ri_numpy - de_ri_jax) / de_ri_numpy.size).round(digit),
            (np.linalg.norm(de_ri_jax - de_ri_torch) / de_ri_numpy.size).round(digit),
            (np.linalg.norm(de_ri_torch - de_ri_numpy) / de_ri_numpy.size).round(digit),]
    res2 = [(np.linalg.norm(de_ti_numpy - de_ti_jax) / de_ti_numpy.size).round(digit),
            (np.linalg.norm(de_ti_jax - de_ti_torch) / de_ti_numpy.size).round(digit),
            (np.linalg.norm(de_ti_torch - de_ti_numpy) / de_ti_numpy.size).round(digit),]
    res3 = [(np.linalg.norm(field_cell_numpy - field_cell_jax) / field_cell_numpy.size).round(digit),
            (np.linalg.norm(field_cell_jax - field_cell_torch) / field_cell_numpy.size).round(digit),
            (np.linalg.norm(field_cell_torch - field_cell_numpy) / field_cell_numpy.size).round(digit),]

    print('Refle', res1)
    print('Trans', res2)
    print('Field', res3)


if __name__ == '__main__':
    option1 = {'pol': 0, 'n_top': 2, 'n_bot': 1, 'theta': 12 * np.pi / 180, 'phi': 0 * np.pi / 180, 'fto': 0,
               'period': [770], 'wavelength': 777, 'thickness': [100], 'fourier_type': 0,
               'ucell': np.array([[[3, 3, 3, 3, 3, 1, 1, 1, 1, 1]], ])}

    option2 = {'pol': 1, 'n_top': 1, 'n_bot': 1.3, 'theta': 0 * np.pi / 180, 'phi': 0 * np.pi / 180, 'fto': 40,
               'period': [2000], 'wavelength': 400, 'thickness': [1000], 'fourier_type': 1,
               'ucell': np.array([[[3, 3, 3.3, 3, 3, 4, 1, 1, 1, 1.2, 1.1, 3, 2, 1.1]], ])}

    option3 = {'psi': 40/180*np.pi, 'n_top': 1, 'n_bot': 1, 'theta': 0 * np.pi / 180, 'phi': 12 * np.pi / 180,
               'fto': 1,
               'period': [200], 'wavelength': 1000, 'thickness': [100], 'fourier_type': 0, 'enhanced_dfs': False,
               'ucell': np.array([[[3, 3, 3.3, 3, 3, 4, 1, 1, 1, 1.2, 1.1, 3, 2, 1.1]], ])}

    option4 = {'psi': 10/180*np.pi, 'n_top': 1, 'n_bot': 1, 'theta': 0 * np.pi / 180, 'phi': 12 * np.pi / 180,
               'fto': [10, 10],
               'period': [200, 600], 'wavelength': 1000, 'thickness': [100, 111, 222, 102, 44], 'fourier_type': 0,
               'enhanced_dfs': True,
               'ucell': np.random.rand(5, 20, 20)*3+1, }

    ucell5 = [
        # layer 1
        [1,[
             ['rectangle', 0+240, 120+240, 160, 80, 4, 0, 0, 0],  # obj 1
             ['rectangle', 0+240, -120+240, 160, 80, 4, 0, 0, 0],  # obj 2
             ['rectangle', 120+240, 0+240, 80, 160, 4, 0, 0, 0],  # obj 3
             ['rectangle', -120+240, 0+240, 80, 160, 4, 0, 0, 0],  # obj 4
         ], ],
    ]

    option5 = {'pol': 0, 'n_top': 2, 'n_bot': 1, 'theta': 12 * np.pi / 180, 'phi': 0 * np.pi / 180, 'fto': 0,
               'period': [770], 'wavelength': 777, 'thickness': [100], 'fourier_type': 0,
               'ucell': ucell5}

    ucell6 = [
        # layer 1
        [3 - 1j, [
             ['rectangle', 0+1000, 410+1000, 160, 80, 4, 0, 0, 0],  # obj 1
             ['ellipse', 0+1000, -10+1000, 160, 80, 4, 1, 20, 20],  # obj 2
             ['rectangle', 120+1000, 500+1000, 80, 160, 4+0.3j, 1.1, 5, 5],  # obj 3
             ['ellipse', -400+1000, -700+1000, 80, 160, 4, 0.4, 20, 20],  # obj 4
         ], ],
        # layer 2
        [3.1, [
             ['rectangle', 0+240, 120+240, 160, 80, 4, 0.4, 5, 5],  # obj 1
             ['ellipse', 0+240, -120+240, 160, 80, 4, 0.1, 20, 20],  # obj 2
             ['ellipse', 120+240, 0+240, 80, 160, 4, 1, 20, 20],  # obj 3
             ['rectangle', -120+240, 0+240, 80, 160, 4, 2, 5, 5],  # obj 4
         ], ],
    ]
    option6 = {'pol': 0, 'n_top': 2, 'n_bot': 1, 'theta': 12 * np.pi / 180, 'phi': 0 * np.pi / 180, 'fto': [5,5],
               'period': [770], 'wavelength': 777, 'thickness': [100, 333], 'fourier_type': 0,
               'ucell': ucell6}

    # consistency_check(option1)
    # consistency_check(option2)
    # consistency_check(option3)
    # consistency_check(option4)

    consistency_check(option5)
    consistency_check(option6)
