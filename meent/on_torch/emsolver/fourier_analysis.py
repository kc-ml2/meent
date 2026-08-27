import torch


def _cfs(x, cell, fto, period, device=torch.device('cpu'), type_complex=torch.complex128):

    cell_next = torch.roll(cell, -1, dims=1)
    cell_diff = cell_next - cell
    cell_diff = cell_diff.type(type_complex)

    modes = torch.arange(-2 * fto, 2 * fto + 1, 1, device=device).type(type_complex)

    center = 2 * fto
    nc = torch.ones(len(modes), device=device).type(torch.bool)

    nc[center] = False

    x_next = torch.vstack((torch.roll(x, -1, dims=0)[:-1], torch.tensor([period], device=device))) - x

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

    if conti_y == 0:
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

    if conti_x == 0:
        conv2d = torch.linalg.inv(conv2d)

    return conv2d


def dfs2d(cell, conti_x, conti_y, fto_x, fto_y, device=torch.device('cpu'), type_complex=torch.complex128):
    cell = cell.type(type_complex)

    ff_x = 2 * fto_x + 1
    ff_y = 2 * fto_y + 1

    cell = cell.T

    if conti_y == 0:
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

    if conti_x == 0:
        conv2d = torch.linalg.inv(conv2d)

    return conv2d


def circulant(fto, device=torch.device('cpu')):
    ff = 2 * fto + 1
    stride = 2 * fto
    circ = torch.zeros((ff, ff), device=device).type(torch.int)
    for r in range(stride + 1):
        idx = torch.arange(-r, -r + ff, 1, device=device)
        circ[r] = idx

    return circ
