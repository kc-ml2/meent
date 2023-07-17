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

        a = torch.hstack([cy - ly / 2, cx - lx / 2])
        b = torch.hstack([cy + ly / 2, cx + lx / 2])

        points = torch.vstack([a, b])

        res = [[points[0], points[1], base]]
        return res

    def rectangle_rotate(self, cx, cy, lx, ly, dx, dy, base, angle=None, angle_margin=1E-5):
        if angle is None:
            angle = torch.tensor(0 * torch.pi / 180)

        # TODO
        if 0 * torch.pi / 2 - angle_margin <= abs(angle) % (2 * torch.pi) <= 0 * torch.pi / 2 + angle_margin:
            return self.rectangle(cx, cy, lx, ly, base), [0, 0], [0, 0]
        elif 1 * torch.pi / 2 - angle_margin <= abs(angle) % (2 * torch.pi) <= 1 * torch.pi / 2 + angle_margin:
            return self.rectangle(cx, cy, ly, lx, base), [0, 0], [0, 0]
        elif 2 * torch.pi / 2 - angle_margin <= abs(angle) % (2 * torch.pi) <= 2 * torch.pi / 2 + angle_margin:
            return self.rectangle(cx, cy, lx, ly, base), [0, 0], [0, 0]
        elif 3 * torch.pi / 2 - angle_margin <= abs(angle) % (2 * torch.pi) <= 3 * torch.pi / 2 + angle_margin:
            return self.rectangle(cx, cy, ly, lx, base), [0, 0], [0, 0]
        else:
            pass

        x_up = (lx / 2).repeat(dx + 2) * torch.linspace(-1, 1, dx + 2)
        y_up = torch.tensor([ly / 2] * (dx + 2))
        x_right = torch.tensor([lx / 2] * (dy + 2))
        y_right = (ly / 2).repeat(dy + 2) * torch.linspace(1, -1, dy + 2)
        # y_right = (ly / 2).repeat(dy + 2) * torch.linspace(-1, 1, dy + 2)
        x_left = torch.tensor([-lx / 2] * (dy + 2))
        y_left = (ly / 2).repeat(dy + 2) * torch.linspace(-1, 1, dy + 2)
        x_down = (lx / 2).repeat(dx + 2) * torch.linspace(1, 1, dx + 2)
        # x_down = (lx / 2).repeat(dx + 2) * torch.linspace(-1, -1, dx + 2)
        y_down = torch.tensor([-ly / 2] * (dx + 2))

        rotate = torch.ones((2, 2), dtype=torch.complex128)
        rotate[0, 0] = torch.cos(angle)
        rotate[0, 1] = -torch.sin(angle)
        rotate[1, 0] = torch.sin(angle)
        rotate[1, 1] = torch.cos(angle)

        UR = rotate @ torch.hstack([lx / 2, ly / 2])
        RD = rotate @ torch.hstack([lx / 2, -ly / 2])
        DL = rotate @ torch.hstack([-lx / 2, -ly / 2])
        LU = rotate @ torch.hstack([-lx / 2, ly / 2])

        UR += torch.tensor([cx, cy])
        RD += torch.tensor([cx, cy])
        DL += torch.tensor([cx, cy])
        LU += torch.tensor([cx, cy])

        p_up = rotate @ torch.vstack((x_up, y_up))
        p_down = rotate @ torch.vstack((x_down, y_down))
        p_left = rotate @ torch.vstack((x_left, y_left))
        p_right = rotate @ torch.vstack((x_right, y_right))

        p_up[0] += cx
        p_up[1] += cy
        # p_up[0], p_up[1] = p_up[0] + cx, p_up[1] + cy
        p_down[0] += cx
        p_down[1] += cy
        p_left[0] += cx
        p_left[1] += cy
        p_right[0] += cx
        p_right[1] += cy

        # TODO: negative angle
        if 0 < (angle / (torch.pi / 180)) % 360 < 90:
            angle_new = (angle % (2 * torch.pi)) - 0

            # trail = L + U
            top1, top4 = UR, DL
            length_1, length_2 = lx, ly
            trail = torch.hstack([p_left, p_up])
            p_out1, p_out2 = p_left, p_up

            if LU[1].real >= RD[1].real:
                top2, top3 = LU, RD
            else:
                top2, top3 = RD, LU

        elif 90 < (angle / (torch.pi / 180)) % 360 < 180:
            angle_new = (angle % (2 * torch.pi)) - torch.pi / 2

            # trail = U + R
            top1, top4 = RD, LU
            length_1, length_2 = ly, lx
            trail = torch.hstack([p_up, p_right])
            p_out1, p_out2 = p_up, p_right

            if UR[1].real >= DL[1].real:
                top2, top3 = UR, DL
            else:
                top2, top3 = DL, UR

        elif 180 < (angle / (torch.pi / 180)) % 360 < 270:
            angle_new = (angle % (2 * torch.pi)) - torch.pi

            # trail = R + D
            top1, top4 = DL, UR
            length_1, length_2 = lx, ly
            trail = torch.hstack([p_right, p_down])
            p_out1, p_out2 = p_right, p_down

            if RD[1].real >= LU[1].real:
                top2, top3 = RD, LU
            else:
                top2, top3 = LU, RD
        elif 270 < (angle / (torch.pi / 180)) % 360 < 360:
            angle_new = (angle % (2 * torch.pi)) - torch.pi * 3 / 2

            # trail = D + L
            top1, top4 = LU, RD
            length_1, length_2 = ly, lx
            trail = torch.hstack([p_down, p_left])
            p_out1, p_out2 = p_down, p_left

            if DL[1].real >= UR[1].real:
                top2, top3 = DL, UR
            else:
                top2, top3 = UR, DL
        else:
            raise ValueError

        obj_list = []

        trail_couple = []
        # trail_couple = torch.zeros(trail.shape, dtype=torch.complex128, requires_grad=False)
        for i, (x, y) in enumerate(zip(*trail)):
            # if i == trail.shape[1]-1:
            #     continue

            if top2[1].real < y.real <= top1[1].real:
                length = ((x - top1[0]) ** 2 + (y - top1[1]) ** 2) ** (1 / 2)
                length = length / torch.cos(angle_new)
                xx = x + length

            elif top3[1].real <= y.real <= top2[1].real:
                if top3[0].real <= top2[0].real:
                    length = length_2 / abs(torch.sin(angle_new))
                # elif top3[0].real > top2[0].real:
                #     length = length_1 / abs(torch.cos(angle_new))
                else:
                    length = length_1 / abs(torch.cos(angle_new))

                # if (x + length).real < trail[0, i+1].real:
                #     xx = trail[0, i+1]
                # else:
                #     xx = x + length
                xx = x + length

            # elif top4[1].real < y.real < top3[1].real:
            else:
                length = ((x - top4[0]) ** 2 + (y - top4[1]) ** 2) ** (1 / 2)

                # TODO: ?? sin and sin?
                # if 0 < (angle / (torch.pi / 180)) % 360 < 90 or 180 < (angle / (torch.pi / 180)) % 360 < 270:
                if 0 < (angle / (torch.pi / 180)) % 360 < 90 or 180 < (angle / (torch.pi / 180)) % 360 < 270:
                    length = length / abs(torch.sin(angle_new))
                else:
                    length = length / abs(torch.sin(angle_new))

                xx = x + length

            # y_next = trail[1, i+1]
            # trail_couple = trail_couple.clone()
            # trail_couple[0, i] = xx
            # trail_couple[1, i] = y
            trail_couple.append([xx, y])

        trail_couple = torch.as_tensor(trail_couple).T
        for index, (x, y) in enumerate(zip(*trail)):
            if index == trail.shape[1] - 1:
                continue

            xx, yy = trail_couple[:, index]
            x_next, y_next = trail[:, index + 1]
            xx_next, yy_next = trail_couple[:, index + 1]
            x_mean = (x_next + x) / 2
            xx_mean = (xx_next + xx) / 2
            # obj_list.append([[x_mean, y], [xx_mean, y_next], base])  # tODO
            obj_list.append([[y, x_mean], [y_next, xx_mean], base])

        return obj_list, p_out1, p_out2

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

        if not row_list or row_list[-1] != self.period[0]:
            row_list.append(self.period[0])
        if not col_list or col_list[-1] != self.period[1]:
            col_list.append(self.period[1])

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
