import os
import time
import argparse
import torch
torch.manual_seed(0)

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from torch.distributions.multivariate_normal import MultivariateNormal

import meent

from meent.on_torch.modeler.modeling import read_material_table, find_nk_index


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('optimizer_index', type=int)

try:
    args = parser.parse_args()
    algo = args.optimizer_index
    print(f'Chosen algorithm index: {algo}')
except:
    algo = 11
    print(f'No input for algorithm index. Index "{algo}" is used')


def plot_spectra(plot_option, wavelength_list, spectra, save_path='', fig_file=''):
    if plot_option == 0:
        return
    if fig_file == '':
        fig_file = str(time.time())

    plt.figure()
    [plt.plot(wavelength_list, spectrum.detach().numpy(), marker='') for spectrum in spectra[:1]]
    if plot_option == 1:
        plt.savefig(save_path + '/' + fig_file + '.jpg')
        plt.close()
    elif plot_option == 2:
        plt.show()
    else:
        raise ValueError


def plot_topview(plot_option, layer_info_list, period, save_path='', fig_file=''):
    if plot_option == 0:
        return
    if fig_file == '':
        fig_file = str(time.time())

    fig, ax = plt.subplots()

    for _, obj_list in layer_info_list:
        for i, obj in enumerate(obj_list):
            xy = (obj[0][1][0].detach().real, obj[0][0][0].detach().real)
            width = abs(obj[1][1][0].detach().real - obj[0][1][0].detach().real)
            height = abs(obj[1][0][0].detach().real - obj[0][0][0].detach().real)
            rec = mpl.patches.Rectangle(xy=xy, width=width, height=height,
                                        angle=0, rotation_point='center', alpha=0.2, facecolor='r')
            ax.add_artist(rec)

    plt.xlim(0, period[0])
    plt.ylim(0, period[1])
    plt.xlabel('X-direction')
    plt.ylabel('Y-direction')
    plt.legend(['User Input', 'Processed Input'])
    if plot_option == 1:
        plt.savefig(save_path + '/' + fig_file + '.jpg')
        plt.close()
    elif plot_option == 2:
        plt.show()
    else:
        raise ValueError


def forward():
    pass


def modelling_ref_index(wavelength, rcwa_options, modeling_options, params_name, params_value, instructions):

    mee = meent.call_mee(wavelength=wavelength, **rcwa_options)

    t = mee.thickness

    for i in range(len(t)):
        if f'l{i+1}_thickness' in params_name:
            t[i] = params_value[params_name[f'l{i+1}_thickness']].reshape((1, 1))
    mee.thickness = t

    mat_table = read_material_table()

    ucell = []
    for i, layer in enumerate(instructions):
        obj_list_per_layer = []
        for j, _ in enumerate(layer):
            instructions_new = []
            instructions_target = instructions[i][j]
            for k, inst in enumerate(instructions_target):
                if k == 0:
                    instructions_new.append(inst)
                elif inst in params_name:
                    instructions_new.append(params_value[params_name[inst]])
                elif inst in modeling_options:
                    if inst[-7:] == 'n_index' and type(modeling_options[inst]) is str:
                        a = find_nk_index(modeling_options[inst], mat_table, wavelength).conj()  # TODO: confirm conj.
                    else:
                        a = modeling_options[inst]
                    instructions_new.append(a)
                else:
                    raise ValueError
            obj_list_per_layer.append(instructions_new)

        a = modeling_options[f'l{i+1}_n_base']
        if type(a) is str:
            a = find_nk_index(a, mat_table, wavelength).conj()

        ucell.append([a, obj_list_per_layer])
    mee.ucell = ucell
    # mee.draw(layer_info_list)

    return mee, ucell


def modelling_ref_index_old(wavelength, rcwa_options, modeling_options, params_name, params_value, instructions):

    mee = meent.call_mee(wavelength=wavelength, **rcwa_options)

    t = mee.thickness

    for i in range(len(t)):
        if f'l{i+1}_thickness' in params_name:
            t[i] = params_value[params_name[f'l{i+1}_thickness']].reshape((1, 1))
    mee.thickness = t

    mat_table = read_material_table()

    layer_info_list = []
    for i, layer in enumerate(instructions):
        obj_list_per_layer = []
        for j, _ in enumerate(layer):
            instructions_new = []
            instructions_target = instructions[i][j]
            for k, inst in enumerate(instructions_target):
                if k == 0:
                    func = getattr(mee, inst)
                elif inst in params_name:
                    instructions_new.append(params_value[params_name[inst]])
                elif inst in modeling_options:
                    if inst[-7:] == 'n_index' and type(modeling_options[inst]) is str:
                        a = find_nk_index(modeling_options[inst], mat_table, wavelength).conj()
                    else:
                        a = modeling_options[inst]
                    instructions_new.append(a)
                else:
                    raise ValueError
            obj_list_per_layer += func(*instructions_new)

        a = modeling_options[f'l{i+1}_n_base']
        if type(a) is str:
            a = find_nk_index(a, mat_table, wavelength).conj()

        layer_info_list.append([a, obj_list_per_layer])

    mee.draw(layer_info_list)

    return mee, layer_info_list


def reflectance_mode_00(mee, wavelength):
    mee.wavelength = wavelength

    de_ri, de_ti = mee.conv_solve()

    x_c, y_c = np.array(de_ti.shape) // 2
    reflectance = de_ri[x_c, y_c]

    return reflectance


def generate_spectrum(rcwa_options, modeling_options, params_name, params_value, instructions, wavelength_list):
    # wavelength_list = rcwa_options['wavelength_list']
    spectrum = torch.zeros(len(wavelength_list))

    for i, wl in enumerate(wavelength_list):
        mee, layer_info_list = modelling_ref_index(wl, rcwa_options, modeling_options, params_name, params_value, instructions)
        spectrum[i] = reflectance_mode_00(mee, wavelength=wl)

    return spectrum, layer_info_list


def gradient_descent(rcwa_options, modeling_options, params_interest, params_gt, instructions, optimizer_option, wavelength_list,
                     n_iters=3, n_steps=3, show_spectrum=0, show_topview=0, algo_name=''):

    gt_name = {k: i for i, (k, v) in enumerate(params_gt.items())}
    gt_value = [v for k, v in params_gt.items()]

    temp_path = f'temp_{str(time.time())}/'
    os.mkdir(temp_path)

    temp_path_spectrum = temp_path + '/spectrum/'
    temp_path_pattern = temp_path + '/pattern/'
    temp_path_loss = temp_path + '/loss/'

    if show_spectrum:
        os.mkdir(temp_path_spectrum)
    if show_topview:
        os.mkdir(temp_path_pattern)

    os.mkdir(temp_path_loss)

    spectrum_gt, layer_info_list_gt = generate_spectrum(rcwa_options, modeling_options, gt_name, gt_value, instructions, wavelength_list)
    
    for ix_iter in range(n_iters):
        pois_name_index, pois_sampled = sampling(params_interest)
        print('initial: ', pois_sampled)
        opt = optimizer_option['optimizer']([pois_sampled], **optimizer_option['options'])

        res_loss_per_iter = torch.zeros(n_steps)

        for ix_step in range(n_steps):
            opt.zero_grad()

            spectrum, layer_info_list = generate_spectrum(rcwa_options, modeling_options, pois_name_index, pois_sampled, instructions, wavelength_list)

            fig_file = str(time.time())
            if show_spectrum:
                plot_spectra(show_spectrum, rcwa_options['wavelength_list'], [spectrum_gt, spectrum],
                             save_path=temp_path_spectrum, fig_file=fig_file)

            if show_topview:
                plot_topview(show_topview, layer_info_list, rcwa_options['period'],
                             save_path=temp_path_pattern, fig_file=fig_file)

            loss = torch.norm(spectrum - spectrum_gt) / spectrum_gt.shape[0]
            loss.backward()
            
            print('loss: ', np.format_float_scientific(loss.detach().numpy(), precision=3),
                  [poi.detach().numpy().round(3) for poi in pois_sampled])
            opt.step()
            
            # save result
            res_loss_per_iter[ix_step] = loss.detach()

        torch.save(res_loss_per_iter, f'{temp_path_loss}_res_loss_all_algo-{algo_name}_{ix_iter}.pt')

    return


def sampling(pois_dist):

    mean = torch.zeros(len(pois_dist))
    std = torch.zeros(len(pois_dist))

    for i, (param_name, (m, s)) in enumerate(pois_dist):
        mean[i], std[i] = m, s

    m = MultivariateNormal(mean, torch.diag(std))

    pois_sampled = m.sample()
    pois_sampled.requires_grad = True
    pois_name_index = {}
    for i, (p_name, _) in enumerate(pois_dist):
        pois_name_index[p_name] = i

    return pois_name_index, pois_sampled


def run(optimizer, n_iters=10, n_steps=50, show_spectrum=0, show_topview=0, algo_name=''):
    rcwa_options = dict(backend=2, thickness=[0, 0, 100000], period=[300, 300], fto=[3, 3],
                        n_top=1, n_bot=1)

    wavelength_list = range(200, 1001, 10)

    modeling_options = dict(
        l1_n_base='sio2',
        l1_o1_angle=20 * torch.pi / 180, l1_o1_c_x=75, l1_o1_c_y=225, l1_o1_n_index='si',
        l1_o1_n_split_x=40, l1_o1_n_split_y=40,
        l1_o2_angle=0 * torch.pi / 180, l1_o2_c_x=225, l1_o2_c_y=75, l1_o2_n_index='si',
        l1_o2_n_split_x=5, l1_o2_n_split_y=5,
        l2_n_base='si3n4',
        l2_o1_length_y=300, l2_o1_angle=0 * torch.pi / 180, l2_o1_c_x=50, l2_o1_c_y=150, l2_o1_n_index='si',
        l2_o1_n_split_x=0, l2_o1_n_split_y=0,
        l2_o2_length_y=300, l2_o2_angle=0 * torch.pi / 180, l2_o2_c_x=200, l2_o2_c_y=150, l2_o2_n_index='si',
        l2_o2_n_split_x=0, l2_o2_n_split_y=0,
        l3_n_base='si'
    )

    # instruction
    instructions = [
        # layer 1
        [
            # obj 1
            ['ellipse', 'l1_o1_c_x', 'l1_o1_c_y', 'l1_o1_length_x', 'l1_o1_length_y', 'l1_o1_n_index', 'l1_o1_angle',
             'l1_o1_n_split_x', 'l1_o1_n_split_y'],
            # obj 2
            ['rectangle', 'l1_o2_c_x', 'l1_o2_c_y', 'l1_o2_length_x', 'l1_o2_length_y', 'l1_o2_n_index',
             'l1_o2_angle', 'l1_o2_n_split_x', 'l1_o2_n_split_y'],
        ],
        # layer 2
        [
            # obj 1
            ['rectangle', 'l2_o1_c_x', 'l2_o1_c_y', 'l2_o1_length_x', 'l2_o1_length_y', 'l2_o1_n_index',
             'l2_o1_angle', 'l2_o1_n_split_x', 'l2_o1_n_split_y'],
            # obj 2
            ['rectangle', 'l2_o2_c_x', 'l2_o2_c_y', 'l2_o2_length_x', 'l2_o2_length_y',
             'l2_o2_n_index', 'l2_o2_angle', 'l2_o2_n_split_x', 'l2_o2_n_split_y'],
        ],
        # layer 3
        [
        ]
    ]

    # parameter of interest
    params_interest = [
        ['l1_o1_length_x', [100, 3]],
        ['l1_o1_length_y', [80, 3]],
        ['l1_o2_length_x', [100, 3]],
        ['l1_o2_length_y', [80, 3]],
        ['l2_o1_length_x', [30, 2]],
        ['l2_o2_length_x', [50, 1]],
        ['l1_thickness', [200, 10]],
        ['l2_thickness', [300, 10]],
    ]

    params_gt = dict(
        l1_o1_length_x=101.5,
        l1_o1_length_y=81.5,
        l1_o2_length_x=98.5,
        l1_o2_length_y=81.5,
        l2_o1_length_x=31,
        l2_o2_length_x=49.5,
        l1_thickness=torch.tensor([205]), l2_thickness=torch.tensor([305]),
    )

    gradient_descent(rcwa_options, modeling_options, params_interest, params_gt, instructions, optimizer, wavelength_list,
                         n_iters=n_iters, n_steps=n_steps, show_spectrum=show_spectrum, show_topview=show_topview, algo_name=algo_name)

    return


if __name__ == '__main__':
    n_iters = 10
    n_steps = 100

    opt10 = {'optimizer': torch.optim.SGD, 'options': {'lr': 1E2, 'momentum': 0.9}}
    opt1a = {'optimizer': torch.optim.SGD, 'options': {'lr': 1E1, 'momentum': 0.9}}
    opt1b = {'optimizer': torch.optim.SGD, 'options': {'lr': 1E0, 'momentum': 0.9}}
    # opt1c = {'optimizer': torch.optim.SGD, 'options': {'lr': 1E-1, 'momentum': 0.9}}
    opt2a = {'optimizer': torch.optim.Adagrad, 'options': {'lr': 1E1}}
    opt2b = {'optimizer': torch.optim.Adagrad, 'options': {'lr': 1E0}}
    opt2c = {'optimizer': torch.optim.Adagrad, 'options': {'lr': 1E-1}}
    opt3a = {'optimizer': torch.optim.RMSprop, 'options': {'lr': 1E1}}
    opt3b = {'optimizer': torch.optim.RMSprop, 'options': {'lr': 1E0}}
    opt3c = {'optimizer': torch.optim.RMSprop, 'options': {'lr': 1E-1}}
    opt4a = {'optimizer': torch.optim.Adam, 'options': {'lr': 1E1}}
    opt4b = {'optimizer': torch.optim.Adam, 'options': {'lr': 1E0}}
    opt4c = {'optimizer': torch.optim.Adam, 'options': {'lr': 1E-1}}
    opt5a = {'optimizer': torch.optim.RAdam, 'options': {'lr': 1E1}}
    opt5b = {'optimizer': torch.optim.RAdam, 'options': {'lr': 1E0}}
    opt5c = {'optimizer': torch.optim.RAdam, 'options': {'lr': 1E-1}}

    optimizers = [opt10, opt1a, opt1b, opt2a, opt2b, opt2c, opt3a, opt3b, opt3c, opt4a, opt4b, opt4c, opt5a, opt5b, opt5c]

    algo_name = optimizers[algo]

    for i, optimizer in enumerate(optimizers[algo:algo+1]):
        file_name = f'{optimizer}_{n_iters}_{n_steps}'
        t0 = time.time()
        run(optimizer, n_iters=n_iters, n_steps=n_steps, show_spectrum=0, show_topview=0, algo_name=algo)

        t1 = time.time()
        print(i, ' run, time: ', t1-t0)

    print(0)
