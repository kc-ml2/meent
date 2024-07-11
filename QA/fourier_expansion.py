import numpy as np


def cfs(x, cell, fto, period, type_complex=np.complex128):

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


def fft_piecewise_constant(cell, x, y, fourier_order_x, fourier_order_y, type_complex=np.complex128):

    period_x, period_y = x[-1], y[-1]

    # X axis
    cell_next_x = np.roll(cell, -1, axis=1)
    cell_diff_x = cell_next_x - cell
    cell_diff_x = cell_diff_x.astype(type_complex)

    cell = cell.astype(type_complex)

    modes_x = np.arange(-2 * fourier_order_x, 2 * fourier_order_x + 1, 1)

    f_coeffs_x = cell_diff_x @ np.exp(-1j * 2 * np.pi * x @ modes_x[None, :] / period_x, dtype=type_complex)
    c = f_coeffs_x.shape[1] // 2

    x_next = np.vstack((np.roll(x, -1, axis=0)[:-1], period_x)) - x

    f_coeffs_x[:, c] = (cell @ np.vstack((x[0], x_next[:-1]))).flatten() / period_x
    mask = np.ones(f_coeffs_x.shape[1], dtype=bool)
    mask[c] = False
    f_coeffs_x[:, mask] /= (1j * 2 * np.pi * modes_x[mask])

    # Y axis
    f_coeffs_x_next_y = np.roll(f_coeffs_x, -1, axis=0)
    f_coeffs_x_diff_y = f_coeffs_x_next_y - f_coeffs_x

    modes_y = np.arange(-2 * fourier_order_y, 2 * fourier_order_y + 1, 1)

    f_coeffs_xy = f_coeffs_x_diff_y.T @ np.exp(-1j * 2 * np.pi * y @ modes_y[None, :] / period_y, dtype=type_complex)
    c = f_coeffs_xy.shape[1] // 2

    y_next = np.vstack((np.roll(y, -1, axis=0)[:-1], period_y)) - y

    f_coeffs_xy[:, c] = f_coeffs_x.T @ np.vstack((y[0], y_next[:-1])).flatten() / period_y

    if c:
        mask = np.ones(f_coeffs_xy.shape[1], dtype=bool)
        mask[c] = False
        f_coeffs_xy[:, mask] /= (1j * 2 * np.pi * modes_y[mask])

    return f_coeffs_xy.T


def cfs2d(cell, x, y, fto_x, fto_y, cx, cy, type_complex=np.complex128):
    cell = cell.astype(type_complex)

    # (cx, cy)
    # (1, 1): epz_conv; (0, 1): epx_conv;  (1, 0): epy_conv

    period_x, period_y = x[-1], y[-1]

    # X axis
    if cx == 0:  # discontinuous in x: inverse rule is applied.
        cell = 1 / cell

    fx = cfs(x, cell, fto_x, period_x)

    if cx == 0:  # discontinuous in x: inverse rule is applied.
        fx = np.linalg.inv(fx)

    # Y axis
    if cy == 0:
        fx = np.linalg.inv(fx)

    fxy = cfs(y, fx.T, fto_y, period_y).T

    if cy == 0:
        fxy = np.linalg.inv(fxy)

    return fxy


def dfs2d(cell, cx, cy, type_complex=np.complex128):
    cell = cell.astype(type_complex)

    # (cx, cy)
    # (1, 1): epz_conv; (0, 1): epx_conv;  (1, 0): epy_conv

    if cx == cy == 1:
        res = np.fft.fft2(ucell/ucell.size).astype(type_complex)

    else:
        rows, cols = cell.shape

        res = np.zeros([rows, cols], dtype=type_complex)

        if cx == 0:  # discontinuous in x: inverse rule is applied.
            cell = 1 / cell

        for r in range(rows):
            res[r, :] = np.fft.fft(cell[r, :] / cols).astype(type_complex)

        if cx == 0:
            res = np.linalg.inv(res)

        if cy == 0:  # discontinuous in y: inverse rule is applied.
            res = np.linalg.inv(res)

        for c in range(cols):
            res[:, c] = np.fft.fft(res[:, c] / rows).astype(type_complex)

        if cy == 0:
            res = np.linalg.inv(res)

    res = np.fft.fftshift(res)

    return res


if __name__ == '__main__':

    ucell = np.array([
            [1, 2, 3, 3, 2],
            [5, 3, 2, 9, 4],
            [1, 3, 6, 4, 1],
            [5, 3, 5, 4, 2],
            [3, 6, 6, 7, 1],
    ])

    f = np.fft.fftshift(np.fft.fft2(ucell/ucell.size))

    a = dfs2d(ucell, 1, 1)
    b = dfs2d(ucell, 1, 0)
    c = dfs2d(ucell, 0, 1)

    x = np.array([1/5, 2/5, 3/5, 4/5, 1]).reshape((-1, 1))
    aa = cfs2d(ucell, x, x, 1, 1, 1, 1)

    aaa = fft_piecewise_constant(ucell, x, x, 1, 1)
    1
