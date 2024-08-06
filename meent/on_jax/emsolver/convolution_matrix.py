import jax
import jax.numpy as jnp

from functools import partial

from .fourier_analysis import dfs2d, cfs2d


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



# @partial(jax.jit, static_argnums=(1,2 ))
# def fft_piecewise_constant(cell, x, y, fto_x, fto_y, type_complex=jnp.complex128):
#
#     period_x, period_y = x[-1], y[-1]
#
#     # X axis
#     cell_next_x = jnp.roll(cell, -1, axis=1)
#     cell_diff_x = cell_next_x - cell
#
#     modes_x = jnp.arange(-2 * fto_x, 2 * fto_x + 1, 1)
#
#     f_coeffs_x = cell_diff_x @ jnp.exp(-1j * 2 * jnp.pi * x @ modes_x[None, :] / period_x).astype(type_complex)
#     c = f_coeffs_x.shape[1] // 2
#
#     x_next = jnp.vstack((jnp.roll(x, -1, axis=0)[:-1], period_x)) - x
#
#     assign_index = (jnp.arange(len(f_coeffs_x)), jnp.array([c]))
#     assign_value = (cell @ jnp.vstack((x[0], x_next[:-1])) / period_x).flatten().astype(type_complex)
#     f_coeffs_x = f_coeffs_x.at[assign_index].set(assign_value)
#
#     mask = jnp.hstack([jnp.arange(c), jnp.arange(c+1, f_coeffs_x.shape[1])])
#     assign_index = mask
#     assign_value = f_coeffs_x[:, mask] / (1j * 2 * jnp.pi * modes_x[mask])
#     f_coeffs_x = f_coeffs_x.at[:, assign_index].set(assign_value)
#
#     # Y axis
#     f_coeffs_x_next_y = jnp.roll(f_coeffs_x, -1, axis=0)
#     f_coeffs_x_diff_y = f_coeffs_x_next_y - f_coeffs_x
#
#     modes_y = jnp.arange(-2 * fto_y, 2 * fto_y + 1, 1)
#
#     f_coeffs_xy = f_coeffs_x_diff_y.T @ jnp.exp(-1j * 2 * jnp.pi * y @ modes_y[None, :] / period_y).astype(type_complex)
#     c = f_coeffs_xy.shape[1] // 2
#
#     y_next = jnp.vstack((jnp.roll(y, -1, axis=0)[:-1], period_y)) - y
#
#     assign_index = [c]
#     assign_value = (f_coeffs_x.T @ jnp.vstack((y[0], y_next[:-1])) / period_y).astype(type_complex)
#     f_coeffs_xy = f_coeffs_xy.at[:, assign_index].set(assign_value)
#
#     if c:
#         mask = jnp.hstack([jnp.arange(c), jnp.arange(c + 1, f_coeffs_x.shape[1])])
#
#         assign_index = mask
#         assign_value = f_coeffs_xy[:, mask] / (1j * 2 * jnp.pi * modes_y[mask])
#
#         f_coeffs_xy = f_coeffs_xy.at[:, assign_index].set(assign_value)
#
#     return f_coeffs_xy.T


def to_conv_mat_vector(ucell_info_list, fto_x, fto_y, device=None,
                       type_complex=jnp.complex128):

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

        # epx_conv_all[i] = epx_conv
        # epy_conv_all[i] = epy_conv
        # epz_conv_i_all[i] = jnp.linalg.inv(epz_conv)

        epx_conv_all = epx_conv_all.at[i].set(epx_conv)
        epy_conv_all = epy_conv_all.at[i].set(epy_conv)
        epz_conv_i_all = epz_conv_i_all.at[i].set(jnp.linalg.inv(epz_conv))

        # f_coeffs = fft_piecewise_constant(ucell_layer, x_list, y_list,
        #                                          fto_x, fto_y, type_complex=type_complex)
        # o_f_coeffs = fft_piecewise_constant(1/ucell_layer, x_list, y_list,
        #                                          fto_x, fto_y, type_complex=type_complex)
        # center = jnp.array(f_coeffs.shape) // 2
        #
        # conv_idx_y = jnp.arange(-ff_y + 1, ff_y, 1)
        # conv_idx_y = circulant(conv_idx_y)
        # conv_i = jnp.repeat(conv_idx_y, ff_x, axis=1)
        # conv_i = jnp.repeat(conv_i, jnp.array([ff_x] * ff_y), axis=0, total_repeat_length=ff_x * ff_y)
        #
        # conv_idx_x = jnp.arange(-ff_x + 1, ff_x, 1)
        # conv_idx_x = circulant(conv_idx_x)
        # conv_j = jnp.tile(conv_idx_x, (ff_y, ff_y))
        #
        # e_conv = f_coeffs[center[0] + conv_i, center[1] + conv_j]
        # o_e_conv = o_f_coeffs[center[0] + conv_i, center[1] + conv_j]
        #
        # e_conv_all = e_conv_all.at[i].set(e_conv)
        # o_e_conv_all = o_e_conv_all.at[i].set(o_e_conv)

    return epx_conv_all, epy_conv_all, epz_conv_i_all


def to_conv_mat_raster_continuous(ucell, fto_x, fto_y, device=None, type_complex=jnp.complex128):

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

        # epx_conv_all[i] = epx_conv
        # epy_conv_all[i] = epy_conv
        # epz_conv_i_all[i] = jnp.linalg.inv(epz_conv)

        epx_conv_all = epx_conv_all.at[i].set(epx_conv)
        epy_conv_all = epy_conv_all.at[i].set(epy_conv)
        epz_conv_i_all = epz_conv_i_all.at[i].set(jnp.linalg.inv(epz_conv))

    return epx_conv_all, epy_conv_all, epz_conv_i_all


# @partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
def to_conv_mat_raster_discrete(ucell, fto_x, fto_y, device=None, type_complex=jnp.complex128,
                                enhanced_dfs=True):
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

        # epx_conv_all[i] = epx_conv
        # epy_conv_all[i] = epy_conv
        # epz_conv_i_all[i] = jnp.linalg.inv(epz_conv)

        epx_conv_all = epx_conv_all.at[i].set(epx_conv)
        epy_conv_all = epy_conv_all.at[i].set(epy_conv)
        epz_conv_i_all = epz_conv_i_all.at[i].set(jnp.linalg.inv(epz_conv))

    return epx_conv_all, epy_conv_all, epz_conv_i_all


def circulant(c):
    center = c.shape[0] // 2
    circ = jnp.zeros((center + 1, center + 1), int)

    for r in range(center+1):
        idx = jnp.arange(r, r - center - 1, -1)
        circ = circ.at[r].set(c[center - idx])

    return circ
