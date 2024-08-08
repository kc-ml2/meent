import torch
# import numpy as np


def _cfs(x, cell, fto, period, device=torch.device('cpu'), type_complex=torch.complex128):

    cell_next = torch.roll(cell, -1, dims=1)
    cell_diff = cell_next - cell
    cell_diff = cell_diff.type(type_complex)

    modes = torch.arange(-2 * fto, 2 * fto + 1, 1, device=device).type(type_complex)

    center = 2 * fto
    # nc = np.ones(len(modes), dtype=bool)
    nc = torch.ones(len(modes), device=device).type(torch.bool)

    nc[center] = False

    # x_next = np.vstack((np.roll(x, -1, axis=0)[:-1], period)) - x
    x_next = torch.vstack((torch.roll(x, -1, dims=0)[:-1], torch.tensor([period], device=device))) - x

    # f = cell_diff @ np.exp(-1j * 2 * np.pi * x @ modes[None, :] / period, dtype=type_complex)
    f = cell_diff @ torch.exp(-1j * 2 * torch.pi * x @ modes[None, :] / period).type(type_complex)

    f[:, nc] /= (1j * 2 * torch.pi * modes[nc])
    f[:, center] = (cell @ torch.vstack((x[0], x_next[:-1]))).flatten() / period

    return f


def cfs2d(cell, x, y, conti_x, conti_y, fto_x, fto_y, device=torch.device('cpu'), type_complex=torch.complex128):
    cell = cell.type(type_complex)
    x = x.type(type_complex)
    y = y.type(type_complex)

    ff_x = 2 * fto_x + 1
    ff_y = 2 * fto_y + 1

    period_x, period_y = x[-1], y[-1]

    cell = cell.T

    if conti_y == 0:  # discontinuous in Y (Row): inverse rule is applied.
        cell = 1 / cell

    cfs1d = _cfs(y, cell, fto_y, period_y, device=device, type_complex=type_complex)

    conv_index_1 = circulant(fto_y, device=device) + (2 * fto_y)
    conv_index_2 = circulant(fto_x, device=device) + (2 * fto_x)

    conv1d = cfs1d[:, conv_index_1]

    if conti_x ^ conti_y:
        conv1d = torch.linalg.inv(conv1d)

    conv1d = conv1d.reshape((-1, ff_y ** 2))

    cfs2d = _cfs(x, conv1d.T, fto_x, period_x, device=device, type_complex=type_complex)

    conv2d = cfs2d[:, conv_index_2]
    conv2d = conv2d.reshape((ff_y, ff_y, ff_x, ff_x))
    conv2d = torch.moveaxis(conv2d, 1, 2)
    conv2d = conv2d.reshape((ff_y*ff_x, ff_y*ff_x))

    if conti_x == 0:  # discontinuous in X (Column): inverse rule is applied.
        conv2d = torch.linalg.inv(conv2d)

    return conv2d


def dfs2d(cell, conti_x, conti_y, fto_x, fto_y, device=torch.device('cpu'), type_complex=torch.complex128):
    cell = cell.type(type_complex)

    ff_x = 2 * fto_x + 1
    ff_y = 2 * fto_y + 1

    cell = cell.T

    if conti_y == 0:  # discontinuous in Y (Row): inverse rule is applied.
        cell = 1 / cell

    dfs1d = torch.fft.fft(cell / cell.shape[1])

    conv_index_1 = circulant(fto_y, device=device)
    conv_index_2 = circulant(fto_x, device=device)

    conv1d_pre = dfs1d[:, conv_index_1]

    if conti_x ^ conti_y:
        conv1d = torch.linalg.inv(conv1d_pre)
    else:
        conv1d = conv1d_pre

    conv1d = conv1d.reshape((-1, ff_y ** 2))

    dfs2d = torch.fft.fft(conv1d.T / conv1d.T.shape[1])

    conv2d = dfs2d[:, conv_index_2]
    conv2d = conv2d.reshape((ff_y, ff_y, ff_x, ff_x))
    conv2d = torch.moveaxis(conv2d, 1, 2)
    conv2d = conv2d.reshape((ff_y*ff_x, ff_y*ff_x))

    if conti_x == 0:  # discontinuous in X (Column): inverse rule is applied.
        conv2d = torch.linalg.inv(conv2d)

    return conv2d


# def dfs2d_debug(cell, conti_x, conti_y, fto_x, fto_y, type_complex=np.complex128, perturbation=1E-10):
#     """
#     algorithm from reticolo.
#     Args:
#         cell:
#         conti_x:
#         conti_y:
#         fto_x:
#         fto_y:
#         type_complex:
#
#     Returns:
#
#     """
#     cell = cell.astype(type_complex)
#
#     ff_x = 2 * fto_x + 1
#     ff_y = 2 * fto_y + 1
#     # fto = max(ff_x, ff_y)
#
#     # (cx, cy)
#     # (1, 1): epz_conv; (0, 1): epx_conv;  (1, 0): epy_conv
#
#     if conti_x == conti_y == 1:
#
#         # case 1
#         fxy = np.fft.fft2(cell/cell.size).astype(type_complex)
#         Y, X = convolution_matrix(fxy, ff_x, ff_y)
#
#         fxy_conv = fxy[Y, X]
#
#         # case 2
#         rows, cols = cell.shape
#         fft1d = np.fft.fft(cell/cell.shape[1]).astype(type_complex)
#         solution = np.fft.fft(fft1d.T/fft1d.shape[0]).T
#
#         conv_index = circulant(fto_y) * 1
#
#         a_conv1d = np.zeros((rows, ff_y, ff_y), dtype=np.complex128)
#
#         for r in range(rows):
#             aa = fft1d[r, conv_index]
#             a_conv1d[r, :, :] = aa
#
#         a_conv1d_reshaped = a_conv1d.reshape(-1, ff_y**2).T
#
#         a_fft2d = np.fft.fft(a_conv1d_reshaped / a_conv1d_reshaped.shape[1])
#
#         a_fft2d_1 = a_fft2d.reshape((3, 3, 6))
#
#         a_conv2d = np.zeros((3, 3, ff_y, ff_y), dtype=np.complex128)
#
#         for r in range(3):
#             for c in range(3):
#                 a_conv2d[:, :, r, c] = a_fft2d_1[r, c, conv_index]
#         a_conv2d_1 = np.moveaxis(a_conv2d, 2, 1)
#         a_conv2d_2 = a_conv2d_1.reshape(ff_y**2, ff_x**2)
#
#         # case 4: RETICOLO
#         bb = np.arange(54).reshape((3,3,6))
#         b_conv2d = np.zeros((3, 3, ff_y, ff_y), dtype=int)
#
#         for r in range(3):
#             for c in range(3):
#                 b_conv2d[:, :, r, c] = bb[r, c, conv_index]
#         b_conv2d_1 = np.moveaxis(b_conv2d, 2, 1)
#         b_conv2d_2 = b_conv2d_1.reshape(ff_y**2, ff_x**2)
#
#         # case 5
#         bb = np.arange(54).reshape((3,3,6))
#         bbb = bb.reshape((9, 6))
#         c_conv2d = np.zeros((6, 3, 3), dtype=int)
#
#         for c in range(bbb.shape[1]):
#             c_conv2d[c] = bbb[:, c].reshape((3, 3))
#
#         c_conv2d_1 = np.block([
#             [c_conv2d[0], c_conv2d[1], c_conv2d[2]],
#             [c_conv2d[-1], c_conv2d[0], c_conv2d[1]],
#             [c_conv2d[-2], c_conv2d[-1], c_conv2d[0]],
#         ])
#
#         # case 5
#         fft1d = np.fft.fft(cell/cell.shape[1]).astype(type_complex)
#
#         axis1_length = fft1d.shape[0]
#         axis2_length = ff_x
#         axis3_length = ff_x
#
#         axis1_coord = np.arange(axis1_length)
#         conv_index_1 = circulant(fto_x)
#         conv_index_2 = circulant(fto_y)
#
#         conv1d = fft1d[:, conv_index_1]
#
#         conv1d_1 = conv1d.reshape((-1, ff_x**2))
#
#         conv1d_2 = conv1d_1[:, np.r_[np.arange(ff_x), np.arange(-ff_x, -1, 1)]]
#
#         conv1d_3 = conv1d_2.T
#         fft2d = np.fft.fft(conv1d_3/conv1d_3.shape[1])
#
#
#
#
#         conv2d = fft2d[:, conv_index_2]
#         conv2d_1 = conv2d.reshape((-1, ff_y**2))
#         conv2d_2 = conv2d_1[:, np.r_[np.arange(ff_y), np.arange(-ff_y, -1, 1)]]
#
#         Y, X = convolution_matrix(conv2d_2, ff_x, ff_y)
#         res = conv2d_2.T[Y, X]
#
#
#
#         fft2d_t = fft2d.T
#         conv2d_t = fft2d_t[conv_index_2, :]
#         conv2d_t_1 = conv2d_t.reshape((ff_y**2, -1))
#         conv2d_t_2 = conv2d_t_1[np.r_[np.arange(ff_y), np.arange(-ff_y, -1, 1)], :]
#
#         conv2d_t_3 = conv2d_t_2
#
#         Y, X = convolution_matrix(conv2d_t_3, ff_x, ff_y)
#         res_t = conv2d_t_3[Y, X]
#
#
#         # case 5
#         bb = np.arange(45).reshape((3,3,5))
#         bbb = bb.reshape((9, 5))
#         bbb = conv2d_1
#         c_conv2da = np.zeros((5, 3, 3), dtype=np.complex128)
#
#         for c in range(bbb.shape[1]):
#             c_conv2da[c] = bbb[:, c].reshape((3, 3))
#
#         c_conv2d_1a = np.block([
#             [c_conv2da[0], c_conv2da[1], c_conv2da[2]],
#             [c_conv2da[-1], c_conv2da[0], c_conv2da[1]],
#             [c_conv2da[-2], c_conv2da[-1], c_conv2da[0]],
#         ])
#
#         Y, X = convolution_matrix(conv2d_2, ff_x, ff_y)
#
#         res = conv2d_2[Y, X]
#
#         # conv2d_1 = conv2d[conv_index_2]
#
#         # case 0
#         center = np.array(bb.shape) // 2
#
#         conv_y = np.arange(-ff_y + 1, ff_y, 1)
#         conv_y = circulant1(conv_y)
#         conv_y = np.repeat(conv_y, ff_x, axis=1)
#         conv_y = np.repeat(conv_y, [ff_x] * ff_y, axis=0)
#
#         conv_x = np.arange(-ff_x + 1, ff_x, 1)
#         conv_x = circulant1(conv_x)
#         conv_x = np.tile(conv_x, (ff_y, ff_y))
#
#
#         # Y, X = convolution_matrix(bb, ff_x, ff_y)
#
#         c = bb[conv_y, conv_x]
#
#         return fxy_conv
#
#     elif conti_x == 1 and conti_y == 0:
#
#         rows, cols = cell.shape
#
#         # o_fy = np.zeros([rows, cols], dtype=type_complex)
#
#         o_cell = 1 / cell  # discontinuous in y: inverse rule is applied.
#
#         # Row direction, Y direction
#         # for c in range(cols):
#         #     # o_fy[:, c] = np.fft.fftshift(np.fft.fft(o_cell[:, c] / rows).astype(type_complex))
#         #     o_fy[:, c] = np.fft.fft(o_cell[:, c] / rows).astype(type_complex)
#
#         o_fy = np.fft.fft(o_cell.T / o_cell.shape[0]).T
#
#
#         idx_conv_y = circulant1(np.arange(-ff_y + 1, ff_y, 1))
#         idx_conv_y1 = circulant(fto_y)
#
#         fy_conv = np.zeros((cols, ff_y, ff_y), dtype=np.complex128)
#
#         for c in range(cols):
#
#             fy_conv[c, :, :] = o_fy[idx_conv_y, c]
#
#         fy_conv = np.linalg.inv(fy_conv)
#
#         fy_conv = fy_conv.reshape(-1, ff_y**2).T
#         # fy_conv = fy_conv.reshape(ff_y**2, -1)
#
#
#         # fxy = np.zeros(fy_conv.shape, dtype=type_complex)
#         #
#         # for r in range(fy_conv.shape[0]):
#         #     # fxy[r, :] = np.fft.fftshift(np.fft.fft(o_fy_conv_i[r, :] / (cols)).astype(type_complex))
#         #     fxy[r, :] = np.fft.fft(fy_conv[r, :] / cols).astype(type_complex)
#
#         fxy = np.fft.fft(fy_conv / fy_conv.shape[1])
#
#         Y, X = convolution_matrix(fxy, ff_x, ff_y)
#
#         fxy_conv = fxy[Y, X]
#
#         return fxy_conv
#
#     elif conti_x == 0 and conti_y == 1:
#
#         rows, cols = cell.shape
#
#         # o_fy = np.zeros([rows, cols], dtype=type_complex)
#
#         # Row direction, Y direction
#         # for c in range(cols):
#         #     # o_fy[:, c] = np.fft.fftshift(np.fft.fft(o_cell[:, c] / rows).astype(type_complex))
#         #     o_fy[:, c] = np.fft.fft(o_cell[:, c] / rows).astype(type_complex)
#
#         o_fy = np.fft.fft(cell.T / cell.shape[0]).T
#
#         idx_conv_y = circulant1(np.arange(-ff_y + 1, ff_y, 1))
#         idx_conv_y1 = circulant(fto_y)
#
#         fy_conv = np.zeros((cols, ff_y, ff_y), dtype=np.complex128)
#
#         for c in range(cols):
#
#             fy_conv[c, :, :] = o_fy[idx_conv_y, c]
#
#         # fy_conv = np.linalg.inv(fy_conv)
#
#         fy_conv = fy_conv.reshape(-1, ff_y**2).T
#
#         # fxy = np.zeros(fy_conv.shape, dtype=type_complex)
#         #
#         # for r in range(fy_conv.shape[0]):
#         #     # fxy[r, :] = np.fft.fftshift(np.fft.fft(o_fy_conv_i[r, :] / (cols)).astype(type_complex))
#         #     fxy[r, :] = np.fft.fft(fy_conv[r, :] / cols).astype(type_complex)
#
#         a = np.where(fy_conv == 0)
#         fy_conv[a] += perturbation
#
#         fxy = np.fft.fft(1/fy_conv / fy_conv.shape[1])
#
#         Y, X = convolution_matrix(fxy, ff_x, ff_y)
#
#         fxy_conv = fxy[Y, X]
#         fxy_conv = np.linalg.inv(fxy_conv)
#
#         return fxy_conv
#
#         #
#         # xx = np.zeros((rows, ff_x, ff_x), dtype=np.complex128)
#         #
#         # for r in range(rows):
#         #
#         #     xx[r, :, :] = fxy[r, a]
#         #
#         # # xxx = np.moveaxis(xx, -1, 0)
#         #
#         # xxx = xx.reshape(-1, ff_y**2)
#         #
#         # conv_x = np.arange(-ff_x + 1, ff_x, 1) + 2
#         # a = circulant(conv_x)
#         #
#         # ff = xxx[a]
#         #
#         #
#         #
#         #
#         # # fff = np.moveaxis(ff, -1, 0)
#         # ffff = ff.reshape(ff_y*ff_x, ff_y*ff_x)
#         #
#         #
#         # fxy = np.fft.fftshift(ff)
#         #
#         #
#         # cx, cy = fxy.shape[0] // 2, fxy.shape[1] // 2
#         #
#         # fxy = fxy[cx - fto:cx + fto + 1, cy - fto:cy + fto + 1]
#         #
#         #
#         #
#         # circ = np.zeros((ff_y, cols//2 + 1), dtype=int)
#         #
#         # for r in range(center + 1):
#         #     idx = np.arange(r, r - center - 1, -1, dtype=int)
#         #
#         #     assign_value = c[center - idx]
#         #     circ[r] = assign_value
#         #
#         #
#         #
#         # conv_y = circulant(conv_y)
#         #
#         # center = c.shape[0] // 2
#         # circ = np.zeros((center + 1, center + 1), dtype=int)
#         #
#         # for r in range(center + 1):
#         #     idx = np.arange(r, r - center - 1, -1, dtype=int)
#         #
#         #     assign_value = c[center - idx]
#         #     circ[r] = assign_value
#         #
#         # return circ
#         #
#         #
#         #
#         #
#         # conv_y = np.repeat(conv_y, ff_y, axis=1)
#         #
#         # conv_y = conv_y.reshape(ff_y, ff_y, 2*cols+1)
#         #
#         # conv_x = np.arange(-cols + 1, cols, 1)
#         # conv_x = circulant(conv_x)
#         # conv_x = np.tile(conv_x, (ff_y, ff_y))
#         #
#         # conv_x = conv_x.reshape(ff_y, ff_y, ff_x)
#         #
#         # o_fy[center[0] + conv_y, center[1] + conv_x]
#         #
#         # o_fy_conv_sub = convolution_matrix(o_fy, ff_x, ff_y)
#         # o_fy_conv_sub_i = np.linalg.inv(o_fy_conv_sub)
#         #
#         #
#         #
#         #
#         # def merge(arr):
#         #     pass
#         #     return arr
#         #
#         # o_fy_conv_i = merge(o_fy_conv_sub_i)
#         #
#         # for r in range(rows):
#         #     fxy[r, :] = np.fft.fft(o_fy_conv_i[r, :] / rows).astype(type_complex)
#         #
#         # fxy = np.fft.fftshift(fxy)
#         # cx, cy = fxy.shape[0] // 2, fxy.shape[1] // 2
#         #
#         # fxy = fxy[cx - fto:cx + fto + 1, cy - fto:cy + fto + 1]
#
#
#     else:
#         rows, cols = cell.shape
#
#         fxy = np.zeros([rows, cols], dtype=type_complex)
#
#         if conti_x == 0:  # discontinuous in x: inverse rule is applied.
#             cell = 1 / cell
#
#         for r in range(rows):
#             fxy[r, :] = np.fft.fft(cell[r, :] / cols).astype(type_complex)
#
#         # if conti_x == 0:
#         #     cx, cy = fxy.shape[0]//2, fxy.shape[1]//2
#         # fxy = fxy[cx-fto:cx+fto+1, cy-fto:cy+fto+1]
#
#         fxy_conv = convolution_matrix(fxy, ff_x, ff_y)
#         fxy_conv_i = np.linalg.inv(fxy_conv)
#
#         # fxy = np.linalg.inv(fxy+np.eye(2*fto+1)*1E-16)
#
#         if conti_y == 0:  # discontinuous in y: inverse rule is applied.
#             cx, cy = fxy.shape[0]//2, fxy.shape[1]//2
#
#             fxy = fxy[cx-fto:cx+fto+1, cy-fto:cy+fto+1]
#
#             fxy = np.linalg.inv(fxy+np.eye(2*fto+1)*1E-16)
#
#         for c in range(fxy.shape[1]):
#             fxy[:, c] = np.fft.fft(fxy[:, c] / rows).astype(type_complex)
#
#         if conti_y == 0:
#             fxy = np.linalg.inv(fxy+np.eye(2*fto+1)*1E-16)
#
#     fxy = np.fft.fftshift(fxy)
#     cx, cy = fxy.shape[0] // 2, fxy.shape[1] // 2
#
#     fxy = fxy[cx - fto:cx + fto + 1, cy - fto:cy + fto + 1]
#
#     return fxy

#
# def convolution_matrix(arr, ff_x, ff_y):
#     center = np.array(arr.shape) // 2
#
#     conv_y = np.arange(-ff_y + 1, ff_y, 1)
#     conv_y = circulant1(conv_y)
#     conv_y = np.repeat(conv_y, ff_x, axis=1)
#     conv_y = np.repeat(conv_y, [ff_x] * ff_y, axis=0)
#
#     conv_x = np.arange(-ff_x + 1, ff_x, 1)
#     conv_x = circulant1(conv_x)
#     conv_x = np.tile(conv_x, (ff_y, ff_y))
#
#     return conv_y, conv_x
#
#
# def circulant1(c):
#     center = c.shape[0] // 2
#     circ = np.zeros((center + 1, center + 1), dtype=int)
#
#     for r in range(center + 1):
#         idx = np.arange(r, r - center - 1, -1, dtype=int)
#
#         assign_value = c[center - idx]
#         circ[r] = assign_value
#
#     return circ
#


def circulant(fto, device=torch.device('cpu')):
    """
    Return circular matrix of indices.
    Args:
        fto: Fourier order, or number of harmonics, in use.
        device:

    Returns: circular matrix of indices.

    """
    ff = 2 * fto + 1
    stride = 2 * fto
    # circ = torch.zeros((ff, ff), device=device, dtype=int)
    circ = torch.zeros((ff, ff), device=device).type(torch.int)
    for r in range(stride + 1):
        idx = torch.arange(-r, -r + ff, 1, device=device)
        circ[r] = idx

    return circ
