import numpy as np


def _cfs(x, cell, fto, period, type_complex=np.complex128):

    cell_next = np.roll(cell, -1, axis=1)
    cell_diff = cell_next - cell

    modes = np.arange(-2 * fto, 2 * fto + 1, 1)

    center = 2 * fto
    nc = np.ones(len(modes), dtype=bool)
    nc[center] = False

    x_next = np.vstack((np.roll(x, -1, axis=0)[:-1], period)) - x

    f = cell_diff @ np.exp(-1j * 2 * np.pi * x @ modes[None, :] / period, dtype=type_complex)

    f[:, nc] /= (1j * 2 * np.pi * modes[nc])
    f[:, center] = (cell @ np.vstack((x[0], x_next[:-1]))).flatten() / period

    return f


def cfs2d(cell, x, y, fto_x, fto_y, cx, cy, type_complex=np.complex128):
    cell = cell.astype(type_complex)

    # (cx, cy)
    # (1, 1): epz_conv; (0, 1): epx_conv;  (1, 0): epy_conv

    period_x, period_y = x[-1], y[-1]

    # X axis
    if cx == 0:  # discontinuous in x: inverse rule is applied.
        cell = 1 / cell

    fx = _cfs(x, cell, fto_x, period_x)

    if cx == 0:  # discontinuous in x: inverse rule is applied.
        fx = np.linalg.inv(fx)

    # Y axis
    if cy == 0:
        fx = np.linalg.inv(fx)

    fxy = _cfs(y, fx.T, fto_y, period_y).T

    if cy == 0:
        fxy = np.linalg.inv(fxy)

    return fxy


def dfs2d(cell, cx, cy, type_complex=np.complex128):
    cell = cell.astype(type_complex)

    # (cx, cy)
    # (1, 1): epz_conv; (0, 1): epx_conv;  (1, 0): epy_conv

    if cx == cy == 1:
        fxy = np.fft.fft2(cell/cell.size).astype(type_complex)

    else:
        rows, cols = cell.shape

        fxy = np.zeros([rows, cols], dtype=type_complex)

        if cx == 0:  # discontinuous in x: inverse rule is applied.
            cell = 1 / cell

        for r in range(rows):
            fxy[r, :] = np.fft.fft(cell[r, :] / cols).astype(type_complex)

        if cx == 0:
            fxy = np.linalg.inv(fxy)

        if cy == 0:  # discontinuous in y: inverse rule is applied.
            fxy = np.linalg.inv(fxy)

        for c in range(cols):
            fxy[:, c] = np.fft.fft(fxy[:, c] / rows).astype(type_complex)

        if cy == 0:
            fxy = np.linalg.inv(fxy)

    fxy = np.fft.fftshift(fxy)

    return fxy
