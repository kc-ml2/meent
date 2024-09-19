import jax
import jax.numpy as jnp

from functools import partial

from .fourier_analysis import dfs2d, cfs2d
from .primitives import meeinv


def cell_compression(cell, type_complex=jnp.complex128):

    cell = jnp.flipud(cell)
    # This is needed because the comp. connecting_algo begins from 0 to period (RC coord. system).
    # On the other hand, the field data is from period to 0 (XY coord. system).
    # Will be flipped again during field reconstruction.

    if type_complex == jnp.complex128:
        type_float = jnp.float64
    else:
        type_float = jnp.float32

    # find discontinuities in x
    step_y, step_x = 1. / jnp.array(cell.shape, dtype=type_float)
    x = []
    y = []
    cell_x = []
    cell_xy = []

    cell_next = jnp.roll(cell, -1, axis=1)

    for col in range(cell.shape[1]):
        if not (cell[:, col] == cell_next[:, col]).all() or (col == cell.shape[1] - 1):
            x.append(step_x * (col + 1))
            cell_x.append(cell[:, col])

    cell_x = jnp.array(cell_x).T
    cell_x_next = jnp.roll(cell_x, -1, axis=0)

    for row in range(cell_x.shape[0]):
        if not (cell_x[row, :] == cell_x_next[row, :]).all() or (row == cell_x.shape[0] - 1):
            y.append(step_y * (row + 1))
            cell_xy.append(cell_x[row, :])

    x = jnp.array(x).reshape((-1, 1))
    y = jnp.array(y).reshape((-1, 1))
    cell_comp = jnp.array(cell_xy)

    return cell_comp, x, y


def to_conv_mat_vector(ucell_info_list, fto_x, fto_y, device=None, type_complex=jnp.complex128, use_pinv=False):

    ff_xy = (2 * fto_x + 1) * (2 * fto_y + 1)

    epx_conv_all = jnp.zeros((len(ucell_info_list), ff_xy, ff_xy)).astype(type_complex)
    epy_conv_all = jnp.zeros((len(ucell_info_list), ff_xy, ff_xy)).astype(type_complex)
    epz_conv_i_all = jnp.zeros((len(ucell_info_list), ff_xy, ff_xy)).astype(type_complex)

    for i, ucell_info in enumerate(ucell_info_list):
        ucell_layer, x_list, y_list = ucell_info
        eps_matrix = ucell_layer ** 2

        epz_conv = cfs2d(eps_matrix, x_list, y_list, 1, 1, fto_x, fto_y, type_complex)
        epy_conv = cfs2d(eps_matrix, x_list, y_list, 1, 0,  fto_x, fto_y, type_complex)
        epx_conv = cfs2d(eps_matrix, x_list, y_list, 0, 1,  fto_x, fto_y, type_complex)

        epx_conv_all = epx_conv_all.at[i].set(epx_conv)
        epy_conv_all = epy_conv_all.at[i].set(epy_conv)
        epz_conv_i_all = epz_conv_i_all.at[i].set(meeinv(epz_conv, use_pinv))

    return epx_conv_all, epy_conv_all, epz_conv_i_all


def to_conv_mat_raster_continuous(ucell, fto_x, fto_y, device=None, type_complex=jnp.complex128, use_pinv=False):

    ff_xy = (2 * fto_x + 1) * (2 * fto_y + 1)

    epx_conv_all = jnp.zeros((ucell.shape[0], ff_xy, ff_xy)).astype(type_complex)
    epy_conv_all = jnp.zeros((ucell.shape[0], ff_xy, ff_xy)).astype(type_complex)
    epz_conv_i_all = jnp.zeros((ucell.shape[0], ff_xy, ff_xy)).astype(type_complex)

    for i, layer in enumerate(ucell):
        n_compressed, x_list, y_list = cell_compression(layer, type_complex=type_complex)
        eps_matrix = n_compressed ** 2

        epz_conv = cfs2d(eps_matrix, x_list, y_list, 1, 1, fto_x, fto_y, type_complex)
        epy_conv = cfs2d(eps_matrix, x_list, y_list, 1, 0, fto_x, fto_y, type_complex)
        epx_conv = cfs2d(eps_matrix, x_list, y_list, 0, 1, fto_x, fto_y, type_complex)

        epx_conv_all = epx_conv_all.at[i].set(epx_conv)
        epy_conv_all = epy_conv_all.at[i].set(epy_conv)
        epz_conv_i_all = epz_conv_i_all.at[i].set(meeinv(epz_conv, use_pinv))

    return epx_conv_all, epy_conv_all, epz_conv_i_all


# @partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
def to_conv_mat_raster_discrete(ucell, fto_x, fto_y, device=None, type_complex=jnp.complex128, enhanced_dfs=True,
                                use_pinv=False):

    ff_xy = (2 * fto_x + 1) * (2 * fto_y + 1)

    epx_conv_all = jnp.zeros((ucell.shape[0], ff_xy, ff_xy)).astype(type_complex)
    epy_conv_all = jnp.zeros((ucell.shape[0], ff_xy, ff_xy)).astype(type_complex)
    epz_conv_i_all = jnp.zeros((ucell.shape[0], ff_xy, ff_xy)).astype(type_complex)

    if enhanced_dfs:
        minimum_pattern_size_y = (4 * fto_y + 1) * ucell.shape[1]
        minimum_pattern_size_x = (4 * fto_x + 1) * ucell.shape[2]
    else:
        minimum_pattern_size_y = 4 * fto_y + 1
        minimum_pattern_size_x = 4 * fto_x + 1
        # e.g., 8 bytes * (40*500) * (40*500) / 1E6 = 3200 MB = 3.2 GB

    for i, layer in enumerate(ucell):
        if layer.shape[0] < minimum_pattern_size_y:
            n = minimum_pattern_size_y // layer.shape[0]
            layer = jnp.repeat(layer, n + 1, axis=0, total_repeat_length=layer.shape[0] * (n + 1))
        if layer.shape[1] < minimum_pattern_size_x:
            n = minimum_pattern_size_x // layer.shape[1]
            layer = jnp.repeat(layer, n + 1, axis=1, total_repeat_length=layer.shape[1] * (n + 1))

        eps_matrix = layer ** 2

        epz_conv = dfs2d(eps_matrix, 1, 1, fto_x, fto_y, type_complex)
        epy_conv = dfs2d(eps_matrix, 1, 0, fto_x, fto_y, type_complex)
        epx_conv = dfs2d(eps_matrix, 0, 1, fto_x, fto_y, type_complex)

        epx_conv_all = epx_conv_all.at[i].set(epx_conv)
        epy_conv_all = epy_conv_all.at[i].set(epy_conv)
        epz_conv_i_all = epz_conv_i_all.at[i].set(meeinv(epz_conv, use_pinv=False))

    return epx_conv_all, epy_conv_all, epz_conv_i_all


def circulant(c):
    center = c.shape[0] // 2
    circ = jnp.zeros((center + 1, center + 1), int)

    for r in range(center+1):
        idx = jnp.arange(r, r - center - 1, -1)
        circ = circ.at[r].set(c[center - idx])

    return circ
