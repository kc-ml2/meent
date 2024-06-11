from typing import Tuple
import random
from glob import glob
from collections import defaultdict
from datetime import datetime
import multiprocessing as mp
from functools import cache
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset


import pytorch_lightning as pl

import meent
# from meent.on_numpy.emsolver.convolution_matrix import to_conv_mat_continuous
from meent.on_numpy.modeler.modeling import find_nk_index, read_material_table

from neuralop.datasets.tensor_dataset import TensorDataset

from threadpoolctl import threadpool_limits, ThreadpoolController
controller = ThreadpoolController()

import constants
import utils

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

def _ts(sample):
    return {'x': sample['permittivity'].unsqueeze(0), 'y': sample['fields']}


# https://github.com/tfp-photonics/neurop_invdes/blob/main/fno_field_prediction/data/fno_unet_data.py
class FieldData(pl.LightningDataModule):
    def __init__(
        self,
        data_path,
        batch_size,
        split,
        cache,
        data_key="design",
        label_key="fields",
        num_workers=4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.save_data_attrs()

    def save_data_attrs(self):
        data = HDF5Dataset(
            self.hparams.data_path / "train",
            data_key=self.hparams.data_key,
            label_key=self.hparams.label_key,
            cache=False,
        )
        attrs = data.attrs
        attrs["out_channels"] = len(data[0][self.hparams.label_key])
        self.save_hyperparameters(attrs)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_data = HDF5Dataset(
                self.hparams.data_path / "train",
                data_key=self.hparams.data_key,
                label_key=self.hparams.label_key,
                cache=self.hparams.cache,
                transform=_ts,
            )
            train, val, _ = random_split(
                train_data,
                (*self.hparams.split, len(train_data) - sum(self.hparams.split)),
            )
            self.train_data = train
            self.val_data = val
        if stage == "test" or stage is None:
            self.test_data = HDF5Dataset(
                self.hparams.data_path / "test",
                data_key=self.hparams.data_key,
                label_key=self.hparams.label_key,
                cache=self.hparams.cache,
                transform=_ts,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.hparams.batch_size)


class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        recursive=False,
        data_key="data",
        label_key="label",
        cache=False,
        transform=None,
    ):
        super().__init__()

        self.data_key = data_key
        self.label_key = label_key
        self._get = self._cached_getitem if cache else self._getitem
        self.cache = cache
        self.transform = transform

        self.attrs = {}
        self.data_keys = []

        p = Path(path)
        if not p.is_dir():
            raise RuntimeError(f"Not a directory: {p}")
        pattern = "**/*.h5" if recursive else "*.h5"
        files = sorted(p.glob(pattern))
        if len(files) < 1:
            raise RuntimeError("No hdf5 datasets found")
        for f in files:
            with h5py.File(f.resolve(), "r") as h5f:
                self.attrs.update(h5f.attrs)
                self.data_keys += [[k, f] for k in h5f.keys() if k != "src"]

    def __getitem__(self, index):
        return self._get(index)

    def _getitem(self, index):
        uid, fp = self.data_keys[index]
        sample = self._from_file(fp, uid)
        if self.transform:
            sample = self.transform(sample)
        return sample

    @cache
    def _cached_getitem(self, index):
        uid, fp = self.data_keys[index]
        sample = self._from_file(fp, uid)
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.data_keys)

    def _from_file(self, fp, uid):
        with h5py.File(fp, "r") as f:
            x = torch.from_numpy(np.array(f[uid][self.data_key], dtype="f4"))
            y = torch.from_numpy(np.array(f[uid][self.label_key], dtype="f4"))
        return {self.data_key: x, self.label_key: y}
#

# yongha
@controller.wrap(limits=4)
def get_field(
        pattern_input, 
        wavelength=1100, #900
        deflected_angle=60, #50
        fourier_order=40,
        field_res=(256, 1, 32) #(64,1,8) 64 x 64 (256,1,32) 256 x 256 # (100, 1, 20) 160 x 100
    ):
    period = [abs(wavelength / np.sin(deflected_angle / 180 * np.pi))]
    n_ridge = 'p_si__real'
    n_groove = 1
    wavelength = np.array([wavelength])
    grating_type = 0
    thickness = [325] * 8

    if type(n_ridge) == str:
        mat_table = read_material_table()
        n_ridge = find_nk_index(n_ridge, mat_table, wavelength)
    ucell = np.array([[pattern_input]])
    ucell = (ucell + 1) / 2
    ucell = ucell * (n_ridge - n_groove) + n_groove
    ucell_new = np.ones((len(thickness), 1, ucell.shape[-1]))
    ucell_new[0:2] = 1.45
    ucell_new[2] = ucell

    mee = meent.call_mee(
        mode=0, wavelength=wavelength, period=period, grating_type=0, n_I=1.45, n_II=1.,
        theta=0, phi=0, psi=0, fourier_order=fourier_order, pol=1,
        thickness=thickness,
        ucell=ucell_new
    )
    # Calculate field distribution: OLD
    de_ri, de_ti, field_cell = mee.conv_solve_field(
        res_x=field_res[0], res_y=field_res[1], res_z=field_res[2],
    )
    if grating_type == 0:
        center = de_ti.shape[0] // 2
        de_ti_cut = de_ti[center - 1:center + 2]
        de_ri_cut = de_ri[center - 1:center + 2][::-1]
    else:
        x_c, y_c = np.array(de_ti.shape) // 2
        de_ti_cut = de_ti[x_c - 1:x_c + 2, y_c - 1:y_c + 2][::-1, ::-1]
        de_ri_cut = de_ri[x_c - 1:x_c + 2, y_c - 1:y_c + 2][::-1, ::-1]

    field_ex= np.flipud(field_cell[:, 0, :, 1])

    return de_ti_cut, field_ex


def gen_struct_uniform(size=10000, width=256):
    l = []
    for i in range(size):
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
    xs = utils.carve_pattern(160, 192, xs)
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

# TODO: copy the main code from beta
if __name__ == "__main__":
    # field_res == (256,1,32) -> [256, 256]
    wavelengths = [900, 1000, 1100]
    wavelength = wavelengths[2] 
    deflected_angles = [50, 60, 70]
    deflected_angle = deflected_angles[1] 

    # (64, 1, 8) 64 x 64 
    # (256, 1, 32) 256 x 256 
    # (100, 1, 20) 160 x 100
    field_res=(256, 1, 32) 

    train_structs = np.load('/data1/EM-data/pirl-structs/20240216_065402/train_structs.npy')
    test_structs = np.load('/data1/EM-data/pirl-structs/20240216_065402/test_structs.npy')
    width = 64
    mfs = field_res[0] // width
    # train_structs = gen_struct_uniform(8000, width=width)
    # train_structs = utils.to_blob(mfs, train_structs)
    # test_structs = gen_struct_uniform(2000, width=width)
    # test_structs = utils.to_blob(mfs, test_structs)

    # for wavelength in wavelengths:
    #     for deflected_angle in deflected_angles:
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

    # f = nn.Upsample(scale_factor=1.5, mode='nearest')