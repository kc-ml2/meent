from bisect import bisect, bisect_left

import torch
import numpy as np

from os import walk
from pathlib import Path


class Compress(torch.autograd.Function):
    """
    Not available as of now and actually no need to use this class.
    https://github.com/pytorch/pytorch/issues/103155
    Seems like a bug that lose gradient information
    """

    @staticmethod
    def setup_context(ctx, inputs, output):
        # layer_info, datatype = inputs
        pass

    @staticmethod
    def forward(layer_info, datatype=torch.complex128):
        pass

    @staticmethod
    def backward(ctx, grad_ucell_layer, grad_x_list, grad_y_list):
        pass


class ModelingTorch:
    def __init__(self, period=None, *args, **kwargs):

        self.ucell = None
        self.ucell_vector = None
        self.x_list = None
        self.y_list = None
        self.mat_table = None
        self.ucell_info_list = None
        self.period = period

    @staticmethod
    def rectangle(cx, cy, lx, ly, base):

        a = torch.hstack([cy - ly / 2, cx - lx / 2])  # row, col
        b = torch.hstack([cy + ly / 2, cx + lx / 2])  # row, col

        res = [[a, b, base]]  # top_left, bottom_right
        return res

    def rectangle_rotate(self, cx, cy, lx, ly, dx, dy, base, angle=None, angle_margin=1E-5):
        ddx, ddy = dx + 2, dy + 2

        if angle is None:
            angle = torch.tensor(0 * torch.pi / 180)

        if 0 * torch.pi / 2 - angle_margin <= abs(angle) % (2 * torch.pi) <= 0 * torch.pi / 2 + angle_margin:
            return self.rectangle(cx, cy, lx, ly, base)
        elif 1 * torch.pi / 2 - angle_margin <= abs(angle) % (2 * torch.pi) <= 1 * torch.pi / 2 + angle_margin:
            return self.rectangle(cx, cy, ly, lx, base)
        elif 2 * torch.pi / 2 - angle_margin <= abs(angle) % (2 * torch.pi) <= 2 * torch.pi / 2 + angle_margin:
            return self.rectangle(cx, cy, lx, ly, base)
        elif 3 * torch.pi / 2 - angle_margin <= abs(angle) % (2 * torch.pi) <= 3 * torch.pi / 2 + angle_margin:
            return self.rectangle(cx, cy, ly, lx, base)
        else:
            pass

        angle = angle % (2 * torch.pi)

        rotate = torch.ones((2, 2), dtype=torch.complex128)
        rotate[0, 0] = torch.cos(angle)
        rotate[0, 1] = -torch.sin(angle)
        rotate[1, 0] = torch.sin(angle)
        rotate[1, 1] = torch.cos(angle)

        UR = rotate @ torch.vstack([lx / 2, ly / 2])
        RD = rotate @ torch.vstack([lx / 2, -ly / 2])
        DL = rotate @ torch.vstack([-lx / 2, -ly / 2])
        LU = rotate @ torch.vstack([-lx / 2, ly / 2])

        UR += torch.tensor([[cx], [cy]])
        RD += torch.tensor([[cx], [cy]])
        DL += torch.tensor([[cx], [cy]])
        LU += torch.tensor([[cx], [cy]])

        if 0 < angle < torch.pi / 2:
            angle_inside = (torch.pi / 2) - angle

            # trail = L + U
            top1, top4 = UR, DL

            if LU[1].real > RD[1].real:
                top2, top3 = LU, RD
                length_top12, length_top24 = lx, ly
                top2_left = True
            else:
                top2, top3 = RD, LU
                length_top12, length_top24 = ly, lx
                top2_left = False

        elif torch.pi / 2 < angle < torch.pi:

            angle_inside = torch.pi - angle
            # trail = U + R
            top1, top4 = RD, LU

            if UR[1].real > DL[1].real:
                top2, top3 = UR, DL
                length_top12, length_top24 = ly, lx
                top2_left = True
            else:
                top2, top3 = DL, UR
                length_top12, length_top24 = lx, ly
                top2_left = False

        elif torch.pi < angle < torch.pi / 2 * 3:
            angle_inside = (torch.pi * 3 / 2) - angle

            # trail = R + D
            top1, top4 = DL, UR

            if RD[1].real > LU[1].real:
                top2, top3 = RD, LU
                length_top12, length_top24 = lx, ly
                top2_left = True
            else:
                top2, top3 = LU, RD
                length_top12, length_top24 = ly, lx
                top2_left = False

        elif torch.pi / 2 * 3 < angle < torch.pi * 2:
            angle_inside = (torch.pi * 2) - angle
            # trail = D + L
            top1, top4 = LU, RD

            if DL[1].real > UR[1].real:
                top2, top3 = DL, UR
                length_top12, length_top24 = ly, lx
                top2_left = True
            else:
                top2, top3 = UR, DL
                length_top12, length_top24 = lx, ly
                top2_left = False
        else:
            raise ValueError

        # point in region 1(top1~top2), 2(top2~top3) and 3(top3~top4)

        xxx, yyy = [], []
        xxx_cp, yyy_cp = [], []
        if top2_left:

            length = length_top12 / torch.sin(angle_inside)
            top3_cp = [top3[0] - length, top3[1]]

            for i in range(ddx + 1):
                x = top1[0] - (top1[0] - top2[0]) / ddx * i
                y = top1[1] - (top1[1] - top2[1]) / ddy * i
                xxx.append(x)
                yyy.append(y)

                xxx_cp.append(x + length / ddx * i)
                yyy_cp.append(y)

            for i in range(ddy + 1):

                x = top2[0] + (top3_cp[0] - top2[0]) / ddx * i
                y = top2[1] - (top2[1] - top3_cp[1]) / ddy * i
                xxx.append(x)
                yyy.append(y)

                xxx_cp.append(x + length)
                yyy_cp.append(y)

            for i in range(ddx + 1):
                x = top3_cp[0] + (top4[0] - top3_cp[0]) / ddx * i
                y = top3_cp[1] - (top3_cp[1] - top4[1]) / ddy * i
                xxx.append(x)
                yyy.append(y)

                xxx_cp.append(x + length / ddx * (ddx - i))
                yyy_cp.append(y)

            obj_list1 = []

            for i in range(len(xxx)):
                if i == len(xxx) - 1:
                    break
                x, y = xxx[i], yyy[i]
                x_cp, y_cp = xxx_cp[i], yyy_cp[i]

                x_next, y_next = xxx[i + 1], yyy[i + 1]
                x_cp_next, y_cp_next = xxx_cp[i + 1], yyy_cp[i + 1]

                x_mean = (x + x_next) / 2
                x_cp_mean = (x_cp + x_cp_next) / 2
                obj_list1.append([[y_cp_next, x_mean], [y, x_cp_mean], base])

            return obj_list1

        else:

            length = length_top12 / torch.cos(angle_inside)
            top3_cp = [top3[0] + length, top3[1]]

            for i in range(ddx + 1):
                x = top1[0] + (top2[0] - top1[0]) / ddx * i
                y = top1[1] - (top1[1] - top2[1]) / ddy * i
                xxx.append(x)
                yyy.append(y)

                xxx_cp.append(x - length / ddx * i)
                yyy_cp.append(y)

            for i in range(ddy + 1):

                x = top2[0] - (top2[0] - top3_cp[0]) / ddx * i
                y = top2[1] - (top2[1] - top3_cp[1]) / ddy * i
                xxx.append(x)
                yyy.append(y)

                xxx_cp.append(x - length)
                yyy_cp.append(y)

            for i in range(ddx + 1):
                x = top3_cp[0] - (top3_cp[0] - top4[0]) / ddx * i
                y = top3_cp[1] - (top3_cp[1] - top4[1]) / ddy * i
                xxx.append(x)
                yyy.append(y)

                xxx_cp.append(x - length / ddx * (ddx - i))
                yyy_cp.append(y)

            obj_list1 = []

            for i in range(len(xxx)):
                if i == len(xxx) - 1:
                    break
                x, y = xxx[i], yyy[i]
                x_cp, y_cp = xxx_cp[i], yyy_cp[i]

                x_next, y_next = xxx[i + 1], yyy[i + 1]
                x_cp_next, y_cp_next = xxx_cp[i + 1], yyy_cp[i + 1]

                x_mean = (x + x_next) / 2
                x_cp_mean = (x_cp + x_cp_next) / 2
                obj_list1.append([[y_cp_next, x_cp_mean], [y, x_mean], base])

            return obj_list1

    def ellipse(self, cx, cy, lx, ly, dx, dy, base, rotation_angle=0 * torch.pi / 180):
        points_x = torch.arange(cx-lx, cx+lx, dx+2)
        points_y = torch.arange(cy-ly, cy+ly, dy+2)

        rotate = torch.tensor([torch.cos(rotation_angle), -torch.sin(rotation_angle)],
                              [torch.sin(rotation_angle),  torch.cos(rotation_angle)],
                              )

        points = rotate @ torch.vstack((points_x, points_y))
        res = [points[0], points[1], base]
        return res

    def vector(self, layer_info, x64=True):

        if x64:
            datatype = torch.complex128
            perturbation = 1E-14
        else:
            datatype = torch.complex64
            perturbation = 1E-6

        pmtvy_base, obj_list = layer_info

        # Griding
        row_list = []
        col_list = []

        for obj in obj_list:
            top_left, bottom_right, _ = obj

            # top_left[0]
            for _ in range(100):
                index = bisect_left(row_list, top_left[0].real, key=lambda x: x.real)
                if len(row_list) > index and top_left[0] == row_list[index]:
                    top_left[0] = top_left[0] - (top_left[0] * perturbation)
                else:
                    row_list.insert(index, top_left[0])
                    break
            else:
                print('WARNING: Overlapping of the objects in modeling is too complicated. Backprop may not work as expected.')
                index = bisect_left(row_list, top_left[0].real, key=lambda x: x.real)
                row_list.insert(index, top_left[0])

            # bottom_right[0]
            for _ in range(100):
                index = bisect_left(row_list, bottom_right[0].real, key=lambda x: x.real)
                if len(row_list) > index and bottom_right[0] == row_list[index]:
                    bottom_right[0] = bottom_right[0] + (bottom_right[0] * perturbation)
                else:
                    row_list.insert(index, bottom_right[0])
                    break
            else:
                print('WARNING: Overlapping of the objects in modeling is too complicated. Backprop may not work as expected.')
                index = bisect_left(row_list, bottom_right[0].real, key=lambda x: x.real)
                row_list.insert(index, bottom_right[0])

            # top_left[1]
            for _ in range(100):
                index = bisect_left(col_list, top_left[1].real, key=lambda x: x.real)
                if len(col_list) > index and top_left[1] == col_list[index]:
                    top_left[1] = top_left[1] - (top_left[1] * perturbation)
                else:
                    col_list.insert(index, top_left[1])
                    break
            else:
                print('WARNING: Overlapping of the objects in modeling is too complicated. Backprop may not work as expected.')
                index = bisect_left(col_list, top_left[1].real, key=lambda x: x.real)
                col_list.insert(index, top_left[1])

            # bottom_right[1]
            for _ in range(100):
                index = bisect_left(col_list, bottom_right[1].real, key=lambda x: x.real)
                if len(col_list) > index and bottom_right[1] == col_list[index]:
                    bottom_right[1] = bottom_right[1] + (bottom_right[1] * perturbation)
                else:
                    col_list.insert(index, bottom_right[1])
                    break
            else:
                print('WARNING: Overlapping of the objects in modeling is too complicated. Backprop may not work as expected.')
                index = bisect_left(col_list, bottom_right[1].real, key=lambda x: x.real)
                col_list.insert(index, bottom_right[1])

        if not row_list or row_list[-1] != self.period[1]:
            row_list.append(self.period[1])
        if not col_list or col_list[-1] != self.period[0]:
            col_list.append(self.period[0])

        if row_list and row_list[0] == 0:
            row_list = row_list[1:]
        if col_list and col_list[0] == 0:
            col_list = col_list[1:]

        ucell_layer = torch.ones((len(row_list), len(col_list)), dtype=datatype, requires_grad=True) * pmtvy_base

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

        x_list = torch.zeros((len(col_list), 1), dtype=datatype)
        y_list = torch.zeros((len(row_list), 1), dtype=datatype)

        for i in range(len(col_list)):
            x_list[i] = col_list[i]

        for i in range(len(row_list)):
            y_list[i] = row_list[i]

        return ucell_layer, x_list, y_list

    def draw(self, layer_info_list):
        ucell_info_list = []

        for layer_info in layer_info_list:
            ucell_layer, x_list, y_list = self.vector(layer_info)
            ucell_info_list.append([ucell_layer, x_list, y_list])
        self.ucell_info_list = ucell_info_list
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
