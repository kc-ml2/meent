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
        self.type_complex = torch.complex128
        # self.type_float = torch.float64

        self.film_layer = None

    def film(self):
        return []

    @staticmethod
    def rectangle_no_approximation(cx, cy, lx, ly, base):

        a = [cy - ly / 2, cx - lx / 2]  # row, col
        b = [cy + ly / 2, cx + lx / 2]  # row, col

        res = [[a, b, base]]  # top_left, bottom_right

        return res

    def rectangle(self, cx, cy, lx, ly, n_index, angle=0, n_split_triangle=2, n_split_parallelogram=2, angle_margin=1E-5):

        if type(lx) in (int, float):
            lx = torch.tensor(lx).reshape(1)
        elif type(lx) is torch.Tensor:
            lx = lx.reshape(1)

        if type(ly) in (int, float):
            ly = torch.tensor(ly).reshape(1)
        elif type(ly) is torch.Tensor:
            ly = ly.reshape(1)

        if type(angle) in (int, float):
            angle = torch.tensor(angle).reshape(1)
        elif type(angle) is torch.Tensor:
            angle = angle.reshape(1)

        lx = lx.type(self.type_float)
        ly = ly.type(self.type_float)
        angle = angle.type(self.type_float)

        angle = angle % (2 * torch.pi)

        # No rotation
        if 0 * torch.pi / 2 - angle_margin <= abs(angle) % (2 * torch.pi) <= 0 * torch.pi / 2 + angle_margin:
            return self.rectangle_no_approximation(cx, cy, lx, ly, n_index)
        elif 1 * torch.pi / 2 - angle_margin <= abs(angle) % (2 * torch.pi) <= 1 * torch.pi / 2 + angle_margin:
            return self.rectangle_no_approximation(cx, cy, ly, lx, n_index)
        elif 2 * torch.pi / 2 - angle_margin <= abs(angle) % (2 * torch.pi) <= 2 * torch.pi / 2 + angle_margin:
            return self.rectangle_no_approximation(cx, cy, lx, ly, n_index)
        elif 3 * torch.pi / 2 - angle_margin <= abs(angle) % (2 * torch.pi) <= 3 * torch.pi / 2 + angle_margin:
            return self.rectangle_no_approximation(cx, cy, ly, lx, n_index)
        else:
            pass

        # Yes rotation
        rotate = torch.ones((2, 2), dtype=self.type_float)
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

        if 0 <= angle < torch.pi / 2:
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

        elif torch.pi / 2 <= angle < torch.pi:

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

        elif torch.pi <= angle < torch.pi / 2 * 3:
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

        elif torch.pi / 2 * 3 <= angle < torch.pi * 2:
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
        if top2_left:

            length = length_top12 / torch.sin(angle_inside)
            top3_cp = [top3[0] - length, top3[1]]

            # 1: Upper triangle
            xxx1 = top1[0] - (top1[0] - top2[0]) / n_split_triangle * torch.arange(n_split_triangle+1).reshape((-1, 1))
            yyy1 = top1[1] - (top1[1] - top2[1]) / n_split_parallelogram * torch.arange(n_split_triangle+1).reshape((-1, 1))
            xxx_cp1 = xxx1 + length / n_split_triangle * torch.arange(n_split_triangle+1).reshape((-1, 1))
            yyy_cp1 = yyy1 * torch.ones(n_split_triangle+1).reshape((-1, 1))

            # 2: Mid parallelogram
            xxx2 = top2[0] + (top3_cp[0] - top2[0]) / n_split_triangle * torch.arange(n_split_parallelogram+1).reshape((-1, 1))
            yyy2 = top2[1] - (top2[1] - top3_cp[1]) / n_split_parallelogram * torch.arange(n_split_parallelogram+1).reshape((-1, 1))
            xxx_cp2 = (xxx2 + length) * torch.ones(n_split_parallelogram+1).reshape((-1, 1))
            yyy_cp2 = yyy2 * torch.ones(n_split_parallelogram+1).reshape((-1, 1))

            # 3: Lower triangle
            xxx3 = top3_cp[0] + (top4[0] - top3_cp[0]) / n_split_triangle * torch.arange(n_split_triangle + 1).reshape(
                (-1, 1))
            yyy3 = top3_cp[1] - (top3_cp[1] - top4[1]) / n_split_parallelogram * torch.arange(n_split_triangle + 1).reshape(
                (-1, 1))

            xxx_cp3 = xxx3 + length / n_split_triangle * torch.arange(n_split_triangle, -1, -1).reshape((-1, 1))
            yyy_cp3 = yyy3 * torch.ones(n_split_triangle + 1).reshape((-1, 1))

            xxx = torch.concat((xxx1, xxx2, xxx3))
            yyy = torch.concat((yyy1, yyy2, yyy3))

            xxx_cp = torch.concat((xxx_cp1, xxx_cp2, xxx_cp3))
            yyy_cp = torch.concat((yyy_cp1, yyy_cp2, yyy_cp3))

            x_mean_arr = (xxx + torch.roll(xxx, -1)) / 2
            x_cp_mean_arr = (xxx_cp + torch.roll(xxx_cp, -1)) / 2
            y_cp_next_arr = torch.roll(yyy_cp, -1)

            obj_list = [[[y_cp_next_arr[i], x_mean_arr[i]], [yyy[i], x_cp_mean_arr[i]], n_index] for i in range(len(xxx)-1)]

        else:
            length = length_top12 / torch.cos(angle_inside)
            top3_cp = [top3[0] + length, top3[1]]

            # 1: Top triangle
            xxx1 = top1[0] + (top2[0] - top1[0]) / n_split_triangle * torch.arange(n_split_triangle + 1).reshape(
                (-1, 1))
            yyy1 = top1[1] - (top1[1] - top2[1]) / n_split_parallelogram * torch.arange(n_split_triangle + 1).reshape(
                (-1, 1))
            xxx_cp1 = xxx1 - length / n_split_triangle * torch.arange(n_split_triangle + 1).reshape((-1, 1))
            yyy_cp1 = yyy1 * torch.ones(n_split_triangle + 1).reshape((-1, 1))

            # 2: Mid parallelogram
            xxx2 = top2[0] - (top2[0] - top3_cp[0]) / n_split_triangle * torch.arange(
                n_split_parallelogram + 1).reshape((-1, 1))
            yyy2 = top2[1] - (top2[1] - top3_cp[1]) / n_split_parallelogram * torch.arange(
                n_split_parallelogram + 1).reshape((-1, 1))
            xxx_cp2 = xxx2 - length * torch.ones(n_split_parallelogram + 1).reshape((-1, 1))
            yyy_cp2 = yyy2 * torch.ones(n_split_parallelogram + 1).reshape((-1, 1))

            # 3: Lower triangle
            xxx3 = top3_cp[0] - (top3_cp[0] - top4[0]) / n_split_triangle * torch.arange(n_split_triangle + 1).reshape(
                (-1, 1))
            yyy3 = top3_cp[1] - (top3_cp[1] - top4[1]) / n_split_parallelogram * torch.arange(
                n_split_triangle + 1).reshape(
                (-1, 1))

            xxx_cp3 = xxx3 - length / n_split_triangle * torch.arange(n_split_triangle, -1, -1).reshape((-1, 1))
            yyy_cp3 = yyy3 * torch.ones(n_split_triangle + 1).reshape((-1, 1))

            xxx = torch.concat((xxx1, xxx2, xxx3))
            yyy = torch.concat((yyy1, yyy2, yyy3))

            xxx_cp = torch.concat((xxx_cp1, xxx_cp2, xxx_cp3))
            yyy_cp = torch.concat((yyy_cp1, yyy_cp2, yyy_cp3))

            x_mean_arr = (xxx + torch.roll(xxx, -1)) / 2
            x_cp_mean_arr = (xxx_cp + torch.roll(xxx_cp, -1)) / 2
            y_cp_next_arr = torch.roll(yyy_cp, -1)

            obj_list = [[[y_cp_next_arr[i], x_cp_mean_arr[i]], [yyy[i], x_mean_arr[i]], n_index] for i in
                         range(len(xxx) - 1)]

        return obj_list

    def ellipse(self, cx, cy, lx, ly, n_index, angle=0, n_split_w=2, n_split_h=2, angle_margin=1E-5, debug=False):

        if type(lx) in (int, float):
            lx = torch.tensor(lx).reshape(1)
        elif type(lx) is torch.Tensor:
            lx = lx.reshape(1)

        if type(ly) in (int, float):
            ly = torch.tensor(ly).reshape(1)
        elif type(ly) is torch.Tensor:
            ly = ly.reshape(1)

        if type(angle) in (int, float):
            angle = torch.tensor(angle).reshape(1)
        elif type(angle) is torch.Tensor:
            angle = angle.reshape(1)

        lx = lx.type(self.type_float)
        ly = ly.type(self.type_float)
        angle = angle.type(self.type_float)

        angle = angle % (2 * torch.pi)

        points_x_origin = lx/2 * torch.cos(torch.linspace(torch.pi/2, 0, n_split_w))
        points_y_origin = ly/2 * torch.sin(torch.linspace(-torch.pi/2, torch.pi/2, n_split_h))

        points_x_origin_contour = lx/2 * torch.cos(torch.linspace(-torch.pi, torch.pi, n_split_w))[:-1]
        points_y_origin_contour = ly/2 * torch.sin(torch.linspace(-torch.pi, torch.pi, n_split_h))[:-1]
        points_origin_contour = torch.vstack([points_x_origin_contour, points_y_origin_contour])

        axis_x_origin = torch.vstack([points_x_origin, torch.ones(len(points_x_origin))])
        axis_y_origin = torch.vstack([torch.ones(len(points_y_origin)), points_y_origin])

        rotate = torch.ones((2, 2), dtype=points_x_origin.dtype)
        rotate[0, 0] = torch.cos(angle)
        rotate[0, 1] = -torch.sin(angle)
        rotate[1, 0] = torch.sin(angle)
        rotate[1, 1] = torch.cos(angle)

        axis_x_origin_rot = rotate @ axis_x_origin
        axis_y_origin_rot = rotate @ axis_y_origin

        axis_x_rot = axis_x_origin_rot[:, :, None]
        axis_x_rot[0] += cx
        axis_x_rot[1] += cy

        axis_y_rot = axis_y_origin_rot[:, :, None]
        axis_y_rot[0] += cx
        axis_y_rot[1] += cy

        points_origin_contour_rot = rotate @ points_origin_contour
        points_contour_rot = points_origin_contour_rot[:, :, None]
        points_contour_rot[0] += cx
        points_contour_rot[1] += cy

        y_highest_index = torch.argmax(points_contour_rot.real, dim=1)[1, 0]

        points_contour_rot = torch.roll(points_contour_rot, (points_contour_rot.shape[1] // 2 - y_highest_index).item(), dims=1)
        y_highest_index = torch.argmax(points_contour_rot.real, dim=1)[1, 0]

        right = points_contour_rot[:, y_highest_index-1]
        left = points_contour_rot[:, y_highest_index+1]

        right_y = right[1].real
        left_y = left[1].real

        left_array = []
        right_array = []

        res = []

        if left_y > right_y:
            right_array.append(points_contour_rot[:, y_highest_index])
        elif left_y < right_y:
            left_array.append(points_contour_rot[:, y_highest_index])

        for i in range(points_contour_rot.shape[1]//2):
            left_array.append(points_contour_rot[:, (y_highest_index+i+1) % points_contour_rot.shape[1]])
            right_array.append(points_contour_rot[:, (y_highest_index-i-1) % points_contour_rot.shape[1]])

        arr = torch.zeros((2, len(right_array) + len(left_array), 1), dtype=points_contour_rot.dtype)

        if left_y > right_y:
            arr[:, ::2] = torch.stack(right_array, dim=1)
            arr[:, 1::2] = torch.stack(left_array, dim=1)
        elif left_y < right_y:
            arr[:, ::2] = torch.stack(left_array, dim=1)
            arr[:, 1::2] = torch.stack(right_array, dim=1)

        arr_roll = torch.roll(arr, -1, 1)

        for i in range(arr.shape[1]):
            ax, ay = arr[:, i]
            bx, by = arr_roll[:, i]

            # LL = [min(ay.real, by.real)+0j, min(ax.real, bx.real)+0j]
            # UR = [max(ay.real, by.real)+0j, max(ax.real, bx.real)+0j]

            LL = [min(ay.real, by.real), min(ax.real, bx.real)]
            UR = [max(ay.real, by.real), max(ax.real, bx.real)]

            res.append([LL, UR, n_index])

        if debug:
            return res[:-1], (axis_x_rot, axis_y_rot, points_contour_rot)
        else:
            return res[:-1]

    def vector_per_layer_numeric(self, layer_info, x64=True):

        if x64:
            datatype = torch.complex128
            perturbation = 0
            perturbation_unit = 1E-14
        else:
            datatype = torch.complex64
            perturbation = 0
            perturbation_unit = 1E-6

        pmtvy_base, obj_list = layer_info

        # Griding
        row_list = []
        col_list = []

        # overlap check and apply perturbation
        for obj in obj_list:
            top_left, bottom_right, _ = obj

            # top_left[0]
            for _ in range(100):
                # index = bisect_left(row_list, top_left[0].real, key=lambda x: x.real)
                index = bisect_left(row_list, top_left[0].real)
                if len(row_list) > index and top_left[0] == row_list[index]:
                    perturbation += perturbation_unit
                    if top_left[0] == 0:
                        top_left[0] = top_left[0] + perturbation

                    else:
                        top_left[0] = top_left[0] + (top_left[0] * perturbation)
                    row_list.insert(index, top_left[0])
                    break
                else:
                    row_list.insert(index, top_left[0])
                    break
            else:
                print('WARNING: Vector modeling has unexpected case. Backprop may not work as expected.')
                # index = bisect_left(row_list, top_left[0].real, key=lambda x: x.real)
                index = bisect_left(row_list, top_left[0].real)
                row_list.insert(index, top_left[0])

            # bottom_right[0]
            for _ in range(100):
                # index = bisect_left(row_list, bottom_right[0].real, key=lambda x: x.real)
                index = bisect_left(row_list, bottom_right[0].real)
                if len(row_list) > index and bottom_right[0] == row_list[index]:
                    perturbation += perturbation_unit
                    bottom_right[0] = bottom_right[0] - (bottom_right[0] * perturbation)
                    row_list.insert(index, bottom_right[0])
                    break

                else:
                    row_list.insert(index, bottom_right[0])
                    break
            else:
                print('WARNING: Vector modeling has unexpected case. Backprop may not work as expected.')
                # index = bisect_left(row_list, bottom_right[0].real, key=lambda x: x.real)
                index = bisect_left(row_list, bottom_right[0].real)
                row_list.insert(index, bottom_right[0])

            # top_left[1]
            for _ in range(100):
                # index = bisect_left(col_list, top_left[1].real, key=lambda x: x.real)
                index = bisect_left(col_list, top_left[1].real)
                if len(col_list) > index and top_left[1] == col_list[index]:
                    perturbation += perturbation_unit

                    if top_left[1] == 0:
                        top_left[1] = top_left[1] + perturbation
                    else:
                        top_left[1] = top_left[1] + (top_left[1] * perturbation)
                    col_list.insert(index, top_left[1])
                    break
                else:
                    col_list.insert(index, top_left[1])
                    break
            else:
                print('WARNING: Vector modeling has unexpected case. Backprop may not work as expected.')
                # index = bisect_left(col_list, top_left[1].real, key=lambda x: x.real)
                index = bisect_left(col_list, top_left[1].real)
                col_list.insert(index, top_left[1])

            # bottom_right[1]
            for _ in range(100):
                # index = bisect_left(col_list, bottom_right[1].real, key=lambda x: x.real)
                index = bisect_left(col_list, bottom_right[1].real)
                if len(col_list) > index and bottom_right[1] == col_list[index]:
                    perturbation += perturbation_unit
                    bottom_right[1] = bottom_right[1] - (bottom_right[1] * perturbation)
                    col_list.insert(index, bottom_right[1])
                    break
                else:
                    col_list.insert(index, bottom_right[1])
                    break
            else:
                print('WARNING: Vector modeling has unexpected case. Backprop may not work as expected.')
                # index = bisect_left(col_list, bottom_right[1].real, key=lambda x: x.real)
                index = bisect_left(col_list, bottom_right[1].real)
                col_list.insert(index, bottom_right[1])

        if not row_list or row_list[-1] != self.period[1]:
            row_list.append(self.period[1].reshape(1).type(datatype))
        if not col_list or col_list[-1] != self.period[0]:
            col_list.append(self.period[0].reshape(1).type(datatype))

        if row_list and row_list[0] == 0:
            row_list = row_list[1:]
        if col_list and col_list[0] == 0:
            col_list = col_list[1:]

        # ucell_layer = torch.ones((len(row_list), len(col_list)), dtype=datatype, requires_grad=True) * pmtvy_base
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

        x_list = torch.cat(col_list).reshape((-1, 1))
        y_list = torch.cat(row_list).reshape((-1, 1))

        return ucell_layer, x_list, y_list

    def draw(self, layer_info_list):
        ucell_info_list = []
        self.film_layer = torch.zeros(len(layer_info_list))

        for i, layer_info in enumerate(layer_info_list):
            ucell_layer, x_list, y_list = self.vector_per_layer_numeric(layer_info)
            ucell_info_list.append([ucell_layer, x_list, y_list])
            if len(x_list) == len(y_list) == 1:
                self.film_layer[i] = 1
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

    # Optimization + Material table
    def modeling_vector_instruction(self, instructions):

        # wavelength = rcwa_options['wavelength']

        # # Thickness update
        # t = rcwa_options['thickness']
        # for i in range(len(t)):
        #     if f'l{i + 1}_thickness' in fitting_parameter_name:
        #         t[i] = fitting_parameter_value[fitting_parameter_name[f'l{i + 1}_thickness']].reshape((1, 1))
        # mee.thickness = t

        # mat_table = read_material_table()

        # Modeling
        layer_info_list = []
        for i, layer in enumerate(instructions):
            obj_list_per_layer = []
            base_refractive_index = layer[0]
            for j, vector_object in enumerate(layer[1]):
                func = getattr(self, vector_object[0])
                obj_list_per_layer += func(*vector_object[1:])

            layer_info_list.append([base_refractive_index, obj_list_per_layer])

        ucell_info_list = self.draw(layer_info_list)

        return ucell_info_list


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
