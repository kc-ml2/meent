import numpy as np

import meent


def consistency_check(option):

    mee = meent.call_mee(backend=0, perturbation=1E-30, **option)  # NumPy
    res_nupmy = mee.conv_solve()
    res_numpy_psi = res_nupmy.res
    res_numpy_te = res_nupmy.res_te_inc
    res_numpy_tm = res_nupmy.res_tm_inc

    field_cell_numpy = mee.calculate_field(res_z=50, res_x=50)

    mee = meent.call_mee(backend=1, perturbation=1E-30, **option)  # JAX
    res_jax = mee.conv_solve()
    res_jax_psi = res_jax.res
    res_jax_te = res_jax.res_te_inc
    res_jax_tm = res_jax.res_tm_inc

    field_cell_jax = mee.calculate_field(res_z=50, res_x=50)

    mee = meent.call_mee(backend=2, perturbation=1E-30, **option)  # PyTorch
    res_torch = mee.conv_solve()
    res_torch_psi = res_torch.res
    res_torch_te = res_torch.res_te_inc
    res_torch_tm = res_torch.res_tm_inc

    field_cell_torch = mee.calculate_field(res_z=50, res_x=50)
    field_cell_torch = field_cell_torch.numpy()

    check_attr = ['R_s', 'R_p', 'T_s', 'T_p', 'de_ri_s', 'de_ri_p', 'de_ri', 'de_ti_s', 'de_ti_p', 'de_ti']

    print('res_psi')
    for attr in check_attr:
        a = getattr(res_numpy_psi, attr)
        b = getattr(res_jax_psi, attr)
        c = getattr(res_torch_psi, attr).numpy()

        res1 = [float(np.linalg.norm(a - b) / a.size),
                float(np.linalg.norm(b - c) / a.size),
                float(np.linalg.norm(c - a) / a.size),]
        print(attr, res1)

    print('res_te')
    for attr in check_attr:
        a = getattr(res_numpy_te, attr)
        b = getattr(res_jax_te, attr)
        c = getattr(res_torch_te, attr).numpy()

        res1 = [float(np.linalg.norm(a - b) / a.size),
                float(np.linalg.norm(b - c) / a.size),
                float(np.linalg.norm(c - a) / a.size),]
        print(attr, res1)

    print('res_tm')
    for attr in check_attr:
        a = getattr(res_numpy_tm, attr)
        b = getattr(res_jax_tm, attr)
        c = getattr(res_torch_tm, attr).numpy()

        res1 = [float(np.linalg.norm(a - b) / a.size),
                float(np.linalg.norm(b - c) / a.size),
                float(np.linalg.norm(c - a) / a.size),]
        print(attr, res1)

    res3 = [float(np.linalg.norm(field_cell_numpy - field_cell_jax) / field_cell_numpy.size),
            float(np.linalg.norm(field_cell_jax - field_cell_torch) / field_cell_numpy.size),
            float(np.linalg.norm(field_cell_torch - field_cell_numpy) / field_cell_numpy.size),]

    print('Field', res3)


if __name__ == '__main__':
    option1 = {'pol': 0, 'n_top': 2, 'n_bot': 1, 'theta': 12 * np.pi / 180, 'phi': 0 * np.pi / 180, 'fto': 0,
               'period': [770], 'wavelength': 777, 'thickness': [100], 'fourier_type': 0,
               'ucell': np.array([[[3, 3, 3, 3, 3, 1, 1, 1, 1, 1]], ])}

    option2 = {'pol': 1, 'n_top': 1, 'n_bot': 1.3, 'theta': 0 * np.pi / 180, 'phi': 0 * np.pi / 180, 'fto': 40,
               'period': [2000], 'wavelength': 400, 'thickness': [1000], 'fourier_type': 1,
               'ucell': np.array([[[3, 3, 3.3, 3, 3, 4, 1, 1, 1, 1.2, 1.1, 3, 2, 1.1]], ])}

    option3 = {'psi': 40/180*np.pi, 'n_top': 1, 'n_bot': 1, 'theta': 0 * np.pi / 180, 'phi': 12 * np.pi / 180,
               'fto': 80,
               'period': [200], 'wavelength': 1000, 'thickness': [100], 'fourier_type': 0, 'enhanced_dfs': False,
               'ucell': np.array([[[3, 3, 3.3, 3, 3, 4, 1, 1, 1, 1.2, 1.1, 3, 2, 1.1]], ])}

    option4 = {'psi': 10/180*np.pi, 'n_top': 1, 'n_bot': 1.5, 'theta': 30 * np.pi / 180, 'phi': 0 * np.pi / 180,
               'fto': [10, 10],
               'period': [200, 600], 'wavelength': 1000, 'thickness': [100, 111, 222, 102, 44], 'fourier_type': 0,
               'enhanced_dfs': True,
               'ucell': np.random.rand(5, 20, 20)*3+1, }

    ucell5 = [
        # layer 1
        [1,[
             ['rectangle', 0+240, 120+240, 160, 80, 4, 1, 20, 20],  # obj 1
             ['rectangle', 0+240, -120+240, 160, 80, 4, 0, 0, 0],  # obj 2
             ['rectangle', 120+240, 0+240, 80, 160, 4, 0, 0, 0],  # obj 3
             ['rectangle', -120+240, 0+240, 80, 160, 4, 0, 0, 0],  # obj 4
         ], ],
    ]

    option5 = {'pol': 0, 'n_top': 2, 'n_bot': 1, 'theta': 12 * np.pi / 180, 'phi': 0 * np.pi / 180, 'fto': [5,5],
               'period': [770], 'wavelength': 777, 'thickness': [100], 'fourier_type': 0,
               'ucell': ucell5}

    ucell6 = [
        # layer 1
        [3, [
             ['rectangle', 0+1000, 410+1000, 160, 80, 4, 0, 0, 0],  # obj 1
             ['ellipse', 0+1000, -10+1000, 160, 80, 4, 1, 20, 20],  # obj 2
             ['rectangle', 120+1000, 500+1000, 80, 160, 4, 1, 5, 5],  # obj 3
             ['ellipse', -400+1000, -700+1000, 80, 160, 4, 0.4, 20, 20],  # obj 4
         ], ],
        # layer 2
        [3.1 - 1j, [
             ['rectangle', 0+240, 120+240, 160, 80, 4, 0, 10, 10],  # obj 1
             ['ellipse', 0+240, -120+240, 160, 80, 4, 0, 20, 20],  # obj 2
             ['ellipse', 120+240, 0+240, 80, 160, 4, 0.4, 20, 20],  # obj 3
             ['rectangle', -120+240, 0+240, 80, 160, 4+3j, 1, 10, 10],  # obj 4
         ], ],
    ]
    option6 = {'pol': 0, 'n_top': 2, 'n_bot': 2, 'theta': 2 * np.pi / 180, 'phi': 10 * np.pi / 180, 'fto': [5, 5],
               'period': [770], 'wavelength': 777, 'thickness': [100, 90], 'fourier_type': 0,
               'ucell': ucell6}

    cands = [option1, option2, option3, option4, option5, option6]
    # cands = [option6]
    for i, case in enumerate(cands):

        print(f'case {i+1}')
        consistency_check(case)
