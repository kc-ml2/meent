import jax.numpy as jnp


def _cfs(x, cell, fto, period, type_complex=jnp.complex128):

    cell_next = jnp.roll(cell, -1, axis=1)
    cell_diff = cell_next - cell

    modes = jnp.arange(-2 * fto, 2 * fto + 1, 1)

    center = 2 * fto
    nc = jnp.ones(len(modes), dtype=bool)
    # nc[center] = False
    nc = nc.at[center].set(False)

    x_next = jnp.vstack((jnp.roll(x, -1, axis=0)[:-1], period)) - x

    # f_coeffs_xy = f_coeffs_x_diff_y.T @ jnp.exp(-1j * 2 * jnp.pi * y @ modes_y[None, :] / period_y).astype(type_complex)
    f = cell_diff @ jnp.exp(-1j * 2 * jnp.pi * x @ modes[None, :] / period).astype(type_complex)

    assign_value = f[:, nc] / (1j * 2 * jnp.pi * modes[nc])
    f = f.at[:, nc].set(assign_value)
    # f[:, nc] /= (1j * 2 * jnp.pi * modes[nc])

    assign_value = (cell @ jnp.vstack((x[0], x_next[:-1]))).flatten() / period
    f = f.at[:, center].set(assign_value)
    # f[:, center] = (cell @ jnp.vstack((x[0], x_next[:-1]))).flatten() / period

    return f


def cfs2d(cell, x, y, conti_x, conti_y, fto_x, fto_y, type_complex=jnp.complex128):
    cell = cell.astype(type_complex)

    ff_x = 2 * fto_x + 1
    ff_y = 2 * fto_y + 1

    period_x, period_y = x[-1], y[-1]

    cell = cell.T

    if conti_y == 0:  # discontinuous in Y (Row): inverse rule is applied.
        cell = 1 / cell

    cfs1d = _cfs(y, cell, fto_y, period_y)

    conv_index_1 = circulant(fto_y) + (2 * fto_y)
    conv_index_2 = circulant(fto_x) + (2 * fto_x)

    conv1d = cfs1d[:, conv_index_1]

    if conti_x ^ conti_y:
        conv1d = jnp.linalg.inv(conv1d)

    conv1d = conv1d.reshape((-1, ff_y ** 2))

    cfs2d = _cfs(x, conv1d.T, fto_x, period_x)

    conv2d = cfs2d[:, conv_index_2]
    conv2d = conv2d.reshape((ff_y, ff_y, ff_x, ff_x))
    conv2d = jnp.moveaxis(conv2d, 1, 2)
    conv2d = conv2d.reshape((ff_y*ff_x, ff_y*ff_x))

    if conti_x == 0:  # discontinuous in X (Column): inverse rule is applied.
        conv2d = jnp.linalg.inv(conv2d)

    return conv2d


def dfs2d(cell, conti_x, conti_y, fto_x, fto_y, type_complex=jnp.complex128):
    cell = cell.astype(type_complex)

    ff_x = 2 * fto_x + 1
    ff_y = 2 * fto_y + 1

    cell = cell.T

    if conti_y == 0:  # discontinuous in Y (Row): inverse rule is applied.
        cell = 1 / cell

    dfs1d = jnp.fft.fft(cell / cell.shape[1])

    conv_index_1 = circulant(fto_y)
    conv_index_2 = circulant(fto_x)

    conv1d = dfs1d[:, conv_index_1]

    if conti_x ^ conti_y:
        conv1d = jnp.linalg.inv(conv1d)

    conv1d = conv1d.reshape((-1, ff_y ** 2))

    dfs2d = jnp.fft.fft(conv1d.T / conv1d.T.shape[1])

    conv2d = dfs2d[:, conv_index_2]
    conv2d = conv2d.reshape((ff_y, ff_y, ff_x, ff_x))
    conv2d = jnp.moveaxis(conv2d, 1, 2)
    conv2d = conv2d.reshape((ff_y*ff_x, ff_y*ff_x))

    if conti_x == 0:  # discontinuous in X (Column): inverse rule is applied.
        conv2d = jnp.linalg.inv(conv2d)

    return conv2d


def circulant(fto):
    """
    Return circular matrix of indices.
    Args:
        fto: Fourier order, or number of harmonics, in use.

    Returns: circular matrix of indices.

    """
    ff = 2 * fto + 1

    stride = 2 * fto

    circ = jnp.zeros((ff, ff), dtype=int)

    for r in range(stride + 1):
        idx = jnp.arange(-r, -r + ff, 1, dtype=int)
        # circ[r] = idx
        circ = circ.at[r].set(idx)

    return circ


# def circulant(c):
#     center = c.shape[0] // 2
#     circ = jnp.zeros((center + 1, center + 1), int)
#
#     for r in range(center+1):
#         idx = jnp.arange(r, r - center - 1, -1)
#         circ = circ.at[r].set(c[center - idx])
#
#     return circ
