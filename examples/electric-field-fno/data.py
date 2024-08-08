import random
import torch
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
from glob import glob
from neuralop.datasets.tensor_dataset import TensorDataset
from threadpoolctl import ThreadpoolController
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

import meent
from meent.on_numpy.modeler.modeling import find_nk_index, read_material_table

import constants
import utils


controller = ThreadpoolController()


class LazyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.xs = glob(f'{path}/x*')
        self.ys = glob(f'{path}/y*')
    
    def __len__(self):
        return len(self.xs)
    
    def __getitem__(self, idx):
        x = torch.from_numpy(np.load(self.xs[idx]))
        y = torch.from_numpy(np.load(self.ys[idx]))

        return {'x': x, 'y': y}                     


@controller.wrap(limits=4)
def get_field(
        pattern_input, 
        wavelength=1100,  # 900
        deflected_angle=60,  # 50
        fto=40,
        field_res=(256, 1, 32)
    ):
    period = [abs(wavelength / np.sin(deflected_angle / 180 * np.pi))]
    n_ridge = 'p_si__real'
    n_groove = 1
    thickness = [325] * 8

    if type(n_ridge) == str:
        mat_table = read_material_table()
        n_ridge = find_nk_index(n_ridge, mat_table, wavelength)
    ucell = pattern_input.numpy().reshape((1, 1, -1))
    ucell = (ucell + 1) / 2
    ucell = ucell * (n_ridge - n_groove) + n_groove
    ucell_new = np.ones((len(thickness), 1, ucell.shape[-1]))
    ucell_new[0:2] = 1.45
    ucell_new[2] = ucell

    mee = meent.call_mee(
        backend=0, wavelength=wavelength, period=period, n_top=1.45, n_bot=1.,
        theta=0, phi=0, psi=0, fto=fto, pol=1,
        thickness=thickness,
        ucell=ucell_new
    )
    # Calculate field distribution: OLD
    de_ri, de_ti, field_cell = mee.conv_solve_field(
        res_x=field_res[0], res_y=field_res[1], res_z=field_res[2],
    )

    field_ex = np.flipud(field_cell[:, 0, :, 1])

    return field_ex


def gen_struct_uniform(size=10000, width=256):
    l = []
    for _ in range(size):
        l.append(
            torch.from_numpy(
                np.array([random.choice([constants.AIR, constants.SILICA]) for _ in range(width)])
            )
        )

    return torch.stack(l)


def generate_data(structs, **kwargs):
    fields = [get_field(struct, **kwargs) for struct in tqdm(structs)]
    xs = np.array(structs)
    xs = xs[:, np.newaxis, np.newaxis, :]
    upper_idx, lower_idx = int(structs.shape[1]*(5/8)), int(structs.shape[1]*(6/8))
    xs = utils.carve_pattern(upper_idx, lower_idx, xs)
    fields = [field[1] for field in fields]
    ys = np.array(fields)
    ys = np.stack([np.real(ys), np.imag(ys)], axis=1)

    return torch.FloatTensor(np.array(xs)), torch.FloatTensor(np.array(ys))


def plot_data(field_ex, field_res, wavelength, deflected_angle):
    plt.imshow(field_ex.real, cmap='jet')
    period = [abs(wavelength / np.sin(deflected_angle / 180 * np.pi))]
    plt.xticks(
        [i for i in np.linspace(0, field_res[0], 2)],
        [int(i * period[0].round(0) / field_res[0]) for i in np.linspace(0, field_res[0], 2)]
    )
    plt.yticks(
        [i for i in np.linspace(0, field_res[2] * 6, 7)][::-1],
        [*[int(325 * (i - 2)) for i in np.linspace(2, 8, 7)]]
    )

    plt.clim(-0.8, 0.8)  # identical to caxis([-4,4]) in MATLAB
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    train_size = 4 # 8000
    test_size = 2 # 2000
    wavelengths = [900, 1000, 1100]
    deflected_angles = [50, 60, 70]

    # (64, 1, 8) 64 x 64 
    # (256, 1, 32) 256 x 256
    # (512, 1, 64) 512 x 512
    field_res=(256, 1, 32) 

    width = 64
    mfs = field_res[0] // width
    train_structs = gen_struct_uniform(train_size, width=width)
    train_structs = utils.to_blob(mfs, train_structs)
    test_structs = gen_struct_uniform(test_size, width=width)
    test_structs = utils.to_blob(mfs, test_structs)

    for wavelength in wavelengths:
        for deflected_angle in deflected_angles:
            train_ds = TensorDataset(
                *generate_data(
                    train_structs,
                    wavelength=wavelength,
                    deflected_angle=deflected_angle,
                    field_res=field_res
                )
            )
            test_ds = TensorDataset(
                *generate_data(
                    test_structs,
                    wavelength=wavelength,
                    deflected_angle=deflected_angle,
                    field_res=field_res
                )
            )

            now = datetime.now().strftime('%M%S')
            torch.save(train_ds, f'{wavelength}-{deflected_angle}-{mfs}-{now}-train-ds.pt')
            torch.save(test_ds, f'{wavelength}-{deflected_angle}-{mfs}-{now}-test-ds.pt')