"""Building a ucell, and looking up the refractive indices that fill it.

Two ways to describe a layer, and this file serves both.

The raster way is an array the caller builds itself; the only help needed is turning material
names into indices, which `put_refractive_index_in_ucell` does.

The vector way describes a layer as shapes - `rectangle`, `ellipse` - on a background. Those
are collected per layer, then `vector_per_layer_numeric` turns them into the (values, x edges,
y edges) triple that `cfs2d` consumes directly, with no raster in between. That is the point of
it: a shape's true edge stays a true edge instead of being rounded onto a grid, and the edges
stay differentiable, so a boundary position can be optimized.

Index convention: meent solves on n - ik, and every lookup here returns that. The tables store
the ordinary optics n + ik, so `find_nk_index` conjugates on the way out.
"""

import warnings
from bisect import bisect, bisect_left

import torch
import numpy as np

from os import walk
from pathlib import Path

from ... import dispersion


def _is_vector_index(n_index):
    if isinstance(n_index, (list, tuple)):
        return len(n_index) == 3
    if torch.is_tensor(n_index):
        return n_index.numel() == 3
    return False


def _to_index_vector(n_index, dtype):
    """Normalise a refractive index to a 3-vector (nx, ny, nz).

    A scalar is broadcast to all three, which is what isotropic means. Elements are converted
    one at a time and stacked rather than passed to `torch.tensor`, so a differentiable index
    keeps its graph - the same reason `_to_tensor` in _base.py is written the way it is.
    """
    if isinstance(n_index, (list, tuple)):
        comps = [c.reshape(()).type(dtype) if torch.is_tensor(c) else torch.tensor(c, dtype=dtype)
                 for c in n_index]
        return torch.stack(comps)
    if torch.is_tensor(n_index):
        if n_index.numel() == 3:
            return n_index.reshape(3).type(dtype)
        return n_index.reshape(()).type(dtype).repeat(3)
    return torch.tensor([n_index, n_index, n_index], dtype=dtype)


class Compress(torch.autograd.Function):
    """Placeholder. Every method is a no-op and nothing calls it.

    Left as a marker for a custom autograd path around the compression step, which is not
    needed as things stand: the compression is written in ordinary tensor operations and
    torch differentiates through it already.
    """

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def forward(layer_info, datatype=torch.complex128):
        pass

    @staticmethod
    def backward(ctx, grad_ucell_layer, grad_x_list, grad_y_list):
        pass


def _insert_or_reuse(coords, value):
    """Return the stored coordinate equal to ``value``, inserting it if it is new.

    ``coords`` stays sorted. A boundary that two rectangles share is ONE edge of the
    piecewise map, so the second rectangle reuses the first one's coordinate instead of
    contributing a second, nearly-coincident one.

    meent used to nudge the duplicate by a relative 1e-14 and insert it anyway. That turns a
    shared boundary into a segment of that width: a 1D cell described as two full-height
    rectangles came out with y edges [1e-14, 0, period, period] - a spurious sliver row plus
    a duplicated one - where the structure is y-invariant and the answer is a single row.
    Measured against the same cell rasterized and Fourier'd analytically, that cost 3e-09 on
    the convolution matrices and 5e-08 on the field, with one bar (no collision) at 6e-12.

    Reusing the tensor is also what the gradient wants: the two rectangles really do share
    that boundary, so moving it has to move both. They stop sharing as soon as the values
    differ, and then no reuse happens.
    """
    index = bisect_left(coords, value.real)
    if len(coords) > index and coords[index] == value:
        return coords[index]
    coords.insert(index, value)
    return value


class ModelingTorch:
    def __init__(self, period=None, *args, **kwargs):

        self.ucell = None
        self.ucell_vector = None
        self.x_list = None
        self.y_list = None
        self.mat_table = None
        self.ucell_info_list = None
        self.period = period

        self.film_layer = None

    def film(self):
        return []

    @staticmethod
    def rectangle_no_approximation(cx, cy, lx, ly, base):

        a = [cy - ly / 2, cx - lx / 2]
        b = [cy + ly / 2, cx + lx / 2]

        res = [[a, b, base]]

        return res

    def rectangle(self, cx, cy, lx, ly, n_index, angle=0, n_split_triangle=2, n_split_parallelogram=2, angle_margin=1E-5):

        if type(lx) in (int, float):
            lx = torch.tensor(lx, dtype=self.type_float).reshape(1)
        elif type(lx) is torch.Tensor:
            lx = lx.reshape(1)

        if type(ly) in (int, float):
            ly = torch.tensor(ly, dtype=self.type_float).reshape(1)
        elif type(ly) is torch.Tensor:
            ly = ly.reshape(1)

        if type(angle) in (int, float):
            angle = torch.tensor(angle, dtype=self.type_float).reshape(1)
        elif type(angle) is torch.Tensor:
            angle = angle.reshape(1)

        lx = lx.type(self.type_float)
        ly = ly.type(self.type_float)
        angle = angle.type(self.type_float)

        angle = angle % (2 * torch.pi)

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

        if top2_left:

            length = length_top12 / torch.sin(angle_inside)
            top3_cp = [top3[0] - length, top3[1]]

            xxx1 = top1[0] - (top1[0] - top2[0]) / n_split_triangle * torch.arange(n_split_triangle+1).reshape((-1, 1))
            yyy1 = top1[1] - (top1[1] - top2[1]) / n_split_parallelogram * torch.arange(n_split_triangle+1).reshape((-1, 1))
            xxx_cp1 = xxx1 + length / n_split_triangle * torch.arange(n_split_triangle+1).reshape((-1, 1))
            yyy_cp1 = yyy1 * torch.ones(n_split_triangle+1).reshape((-1, 1))

            xxx2 = top2[0] + (top3_cp[0] - top2[0]) / n_split_triangle * torch.arange(n_split_parallelogram+1).reshape((-1, 1))
            yyy2 = top2[1] - (top2[1] - top3_cp[1]) / n_split_parallelogram * torch.arange(n_split_parallelogram+1).reshape((-1, 1))
            xxx_cp2 = (xxx2 + length) * torch.ones(n_split_parallelogram+1).reshape((-1, 1))
            yyy_cp2 = yyy2 * torch.ones(n_split_parallelogram+1).reshape((-1, 1))

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

            xxx1 = top1[0] + (top2[0] - top1[0]) / n_split_triangle * torch.arange(n_split_triangle + 1).reshape(
                (-1, 1))
            yyy1 = top1[1] - (top1[1] - top2[1]) / n_split_parallelogram * torch.arange(n_split_triangle + 1).reshape(
                (-1, 1))
            xxx_cp1 = xxx1 - length / n_split_triangle * torch.arange(n_split_triangle + 1).reshape((-1, 1))
            yyy_cp1 = yyy1 * torch.ones(n_split_triangle + 1).reshape((-1, 1))

            xxx2 = top2[0] - (top2[0] - top3_cp[0]) / n_split_triangle * torch.arange(
                n_split_parallelogram + 1).reshape((-1, 1))
            yyy2 = top2[1] - (top2[1] - top3_cp[1]) / n_split_parallelogram * torch.arange(
                n_split_parallelogram + 1).reshape((-1, 1))
            xxx_cp2 = xxx2 - length * torch.ones(n_split_parallelogram + 1).reshape((-1, 1))
            yyy_cp2 = yyy2 * torch.ones(n_split_parallelogram + 1).reshape((-1, 1))

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
            lx = torch.tensor(lx, dtype=self.type_float).reshape(1)
        elif type(lx) is torch.Tensor:
            lx = lx.reshape(1)

        if type(ly) in (int, float):
            ly = torch.tensor(ly, dtype=self.type_float).reshape(1)
        elif type(ly) is torch.Tensor:
            ly = ly.reshape(1)

        if type(angle) in (int, float):
            angle = torch.tensor(angle, dtype=self.type_float).reshape(1)
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


            LL = [min(ay.real, by.real), min(ax.real, bx.real)]
            UR = [max(ay.real, by.real), max(ax.real, bx.real)]

            res.append([LL, UR, n_index])

        if debug:
            return res[:-1], (axis_x_rot, axis_y_rot, points_contour_rot)
        else:
            return res[:-1]

    def vector_per_layer_numeric(self, layer_info, x64=True):

        datatype = torch.complex128 if x64 else torch.complex64

        pmtvy_base, obj_list = layer_info

        row_list = []
        col_list = []

        for obj in obj_list:
            top_left, bottom_right, _ = obj

            top_left[0] = _insert_or_reuse(row_list, top_left[0])

            bottom_right[0] = _insert_or_reuse(row_list, bottom_right[0])

            top_left[1] = _insert_or_reuse(col_list, top_left[1])
            bottom_right[1] = _insert_or_reuse(col_list, bottom_right[1])

        if not row_list or row_list[-1] != self.period[1]:
            row_list.append(self.period[1].reshape(1).type(datatype))
        if not col_list or col_list[-1] != self.period[0]:
            col_list.append(self.period[0].reshape(1).type(datatype))

        if row_list and row_list[0] == 0:
            row_list = row_list[1:]
        if col_list and col_list[0] == 0:
            col_list = col_list[1:]

        is_aniso = _is_vector_index(pmtvy_base) or any(_is_vector_index(obj[2]) for obj in obj_list)

        if is_aniso:
            base = _to_index_vector(pmtvy_base, datatype)
            ucell_layer = torch.ones((len(row_list), len(col_list), 3), dtype=datatype) * base
        else:
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

            if is_aniso:
                ucell_layer[row_begin:row_end, col_begin:col_end, :] = _to_index_vector(pmty, datatype)
            else:
                ucell_layer[row_begin:row_end, col_begin:col_end] = pmty

        x_list = torch.cat(col_list).reshape((-1, 1))
        y_list = torch.cat(row_list).reshape((-1, 1))

        ucell_layer = ucell_layer.to(self.device)
        x_list = x_list.to(self.device)
        y_list = y_list.to(self.device)

        return ucell_layer, x_list, y_list

    def draw(self, layer_info_list):
        """Convert every layer's shape list into its (values, x edges, y edges) triple.

        `film_layer` marks layers that came out as a single cell - uniform, with no internal
        boundary. Those have no structure to diffract from, and knowing so lets the eigenvalue
        problem for them be skipped.
        """
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
        """Replace material indices in a raster ucell with refractive indices at wavelength wl.

        The incoming ucell holds integers indexing into `mat_list`, so one array describes the
        structure at every wavelength and only this step is repeated across a sweep. An entry
        may be a name to look up, a dispersion spec, or a number to use as it stands.
        """
        res = torch.zeros(ucell.shape, device=device, dtype=type_complex)
        ucell_mask = torch.tensor(ucell, device=device, dtype=type_complex)
        for i_mat, material in enumerate(mat_list):
            mask = torch.nonzero(ucell_mask == i_mat, as_tuple=True)

            if isinstance(material, (str, dict)):
                if isinstance(material, str) and not self.mat_table:
                    self.mat_table = read_material_table()
                assign_value = find_nk_index(material, self.mat_table, wl)
            else:
                assign_value = material
            res[mask] = assign_value

        return res

    def modeling_vector_instruction(self, instructions):
        """Run a nested instruction list into layer descriptions.

        Each entry is [background index, [[shape name, *args], ...]], and the shape name is
        resolved against this object by `getattr` - so `rectangle` and `ellipse` are reachable
        by name, and adding a method adds an instruction with no dispatch table to update.
        """


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


def warn_if_out_of_range(material, mat_data, wl):
    try:
        wl_min, wl_max = float(np.min(wl)), float(np.max(wl))
    except Exception:
        return

    if isinstance(mat_data, dict):
        low, high = mat_data['wavelength_range']
    else:
        low, high = float(np.min(mat_data[:, 0])), float(np.max(mat_data[:, 0]))
    if wl_min < low or wl_max > high:
        warnings.warn(
            f'{material}: wavelength {wl_min:.4e} - {wl_max:.4e} falls outside the tabulated '
            f'range {low:.4e} - {high:.4e}; values are clamped to the endpoints. '
            f'Check that the wavelength unit matches the table.',
            stacklevel=3)


def find_nk_index(material, mat_table, wl):
    """Refractive index of one material at wavelength wl, as n - ik.

    Three kinds of input: a dict is a dispersion spec evaluated on the spot, a name ending in
    `__real` asks for n alone with absorption dropped, and any other name is looked up in the
    table and interpolated.

    Both routes end in a conjugation, explicit here and as the `-` on the k term below: the
    tables and the formulas are quoted on the ordinary optics convention n + ik, and meent
    solves on n - ik. Getting this backwards turns an absorbing material into a gain medium,
    which does not raise anything - it returns more power than went in.
    """
    if isinstance(material, dict):
        mat_data = material
        material_name = mat_data.get('formula', 'dynamic material')
        n_only = False
    elif material[-6:] == '__real':
        material = material[:-6]
        n_only = True
        mat_data = mat_table[material.upper()]
        material_name = material
    else:
        n_only = False
        mat_data = mat_table[material.upper()]
        material_name = material
    warn_if_out_of_range(material_name, mat_data, wl)

    if isinstance(mat_data, dict):
        n_index = dispersion.evaluate(mat_data, wl, np)
        if n_only:
            return np.real(n_index)
        return np.conj(n_index + 0j)

    n_index = np.interp(wl, mat_data[:, 0], mat_data[:, 1])

    if n_only:
        return n_index

    k_index = np.interp(wl, mat_data[:, 0], mat_data[:, 2])
    nk = n_index - 1j * k_index

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
            spec = dispersion.parse_header(path)
            if spec is not None:
                mat_table[name[:-4].upper()] = spec
                continue
            data = np.loadtxt(path, skiprows=1)
            mat_table[name[:-4].upper()] = data.astype(type_complex)

        elif name[-3:] == 'mat':
            from scipy.io import loadmat
            data = loadmat(path)
            data = np.array([data['WL'], data['n'], data['k']], dtype=type_complex)[:, :, 0].T
            mat_table[name[:-4].upper()] = data
    return mat_table
