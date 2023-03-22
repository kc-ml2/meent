import torch
import numpy as np

from os import walk
from pathlib import Path


class ModelingTorch:
    def __init__(self, *args, **kwargs):

        self.ucell = None
        self.ucell_vector = None
        self.x_list = None
        self.y_list = None
        self.mat_table = None

    def vector(self, layer_info, datatype=torch.complex128):
        period, pmtvy_base, obj_list = layer_info

        # Griding
        row_list = []
        col_list = []

        for obj in obj_list:
            top_left, bottom_right, pmty = obj
            row_list.extend([top_left[0], bottom_right[0]])
            col_list.extend([top_left[1], bottom_right[1]])

        row_list = list(set(row_list))
        col_list = list(set(col_list))

        row_list.sort()
        col_list.sort()

        if not row_list or row_list[-1] != period[0]:
            row_list.append(period[0])
        if not col_list or col_list[-1] != period[1]:
            col_list.append(period[1])

        if row_list and row_list[0] == 0:
            row_list = row_list[1:]
        if col_list and col_list[0] == 0:
            col_list = col_list[1:]

        ucell_layer = torch.ones((len(row_list), len(col_list)), dtype=datatype) * pmtvy_base

        for obj in obj_list:
            top_left, bottom_right, pmty = obj
            if top_left[0] == 0:
                row_begin = 0
            else:
                row_begin = row_list.index(top_left[0]) + 1
            row_end = row_list.index(bottom_right[0]) + 1

            if top_left[1] == 0:
                col_begin = 0
            else:
                col_begin = col_list.index(top_left[1]) + 1
            col_end = col_list.index(bottom_right[1]) + 1

            ucell_layer[row_begin:row_end, col_begin:col_end] = pmty

        x_list = torch.tensor(col_list, dtype=datatype).reshape((-1, 1)) / period[0]
        y_list = torch.tensor(row_list, dtype=datatype).reshape((-1, 1)) / period[1]

        return ucell_layer, x_list, y_list

    def draw(self, layer_info_list):
        ucell_info_list = []

        for layer_info in layer_info_list:
            ucell_layer, x_list, y_list = self.vector(layer_info)
            ucell_info_list.append([ucell_layer, x_list, y_list])

        return ucell_info_list

    def put_refractive_index_in_ucell(self, ucell, mat_list, wl, device=torch.device('cpu'), type_complex=torch.complex128):
        res = torch.zeros(ucell.shape, device=device, dtype=type_complex)
        ucell_mask = torch.tensor(ucell, device=device, dtype=type_complex)
        for i_mat, material in enumerate(mat_list):
            mask = torch.nonzero(ucell_mask == i_mat, as_tuple=True)

            if type(material) == str:
                if not self.mat_table:
                    self.mat_table = read_material_table()
                assign_value = find_nk_index(material, self.mat_table, wl)
            else:
                assign_value = material
            res[mask] = assign_value

        return res


# def put_permittivity_in_ucell_object(ucell_size, mat_list, obj_list, mat_table, wl, device=torch.device('cpu'),
#                                      type_complex=torch.complex128):
#     """
#     Under development
#     """
#     res = torch.zeros(ucell_size, device=device).type(type_complex)
#
#     for material, obj_index in zip(mat_list, obj_list):
#         if type(material) == str:
#             res[obj_index] = find_nk_index(material, mat_table, wl) ** 2
#         else:
#             res[obj_index] = material ** 2
#
#     return res


def find_nk_index(material, mat_table, wl):
    if material[-6:] == '__real':
        material = material[:-6]
        n_only = True
    else:
        n_only = False

    mat_data = mat_table[material.upper()]
    n_index = np.interp(wl, mat_data[:, 0], mat_data[:, 1])

    if n_only:
        return n_index

    k_index = np.interp(wl, mat_data[:, 0], mat_data[:, 2])
    nk = n_index + 1j * k_index

    return nk


def read_material_table(nk_path=None, type_complex=torch.complex128):
    if type_complex == torch.complex128:
        type_complex = np.float64
    elif type_complex == torch.complex64:
        type_complex = np.float32
    else:
        raise ValueError

    mat_table = {}

    if nk_path is None:
        nk_path = str(Path(__file__).resolve().parent.parent.parent) + '/nk_data'

    full_path_list, name_list, _ = [], [], []
    for (dirpath, dirnames, filenames) in walk(nk_path):
        full_path_list.extend([f'{dirpath}/{filename}' for filename in filenames])
        name_list.extend(filenames)
    for path, name in zip(full_path_list, name_list):
        if name[-3:] == 'txt':
            data = np.loadtxt(path, skiprows=1)
            mat_table[name[:-4].upper()] = data.astype(type_complex)

        elif name[-3:] == 'mat':
            from scipy.io import loadmat
            data = loadmat(path)
            data = np.array([data['WL'], data['n'], data['k']], dtype=type_complex)[:, :, 0].T
            mat_table[name[:-4].upper()] = data
    return mat_table
