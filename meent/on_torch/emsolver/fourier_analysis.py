"""Fourier coefficients of a permittivity layer, and the convolution matrices built from them.

Two routes to the same object. `cfs2d` integrates a piecewise-constant description exactly and
wants explicit edge coordinates; `dfs2d` runs an FFT over a uniform raster and wants none. They
are `fourier_type` 1 and 0 respectively.

Both take `conti_x` / `conti_y`, which say whether the field component this matrix will multiply
is continuous across a discontinuity in that direction. That is Li's factorization: a product of
two functions that jump at the same place converges to the wrong limit under the direct rule, and
the fix is to expand the reciprocal and invert. 1 selects the direct rule, 0 the inverse rule. The
caller picks the pair per component - see `to_conv_mat_raster_*` in convolution_matrix.py.
"""

import torch


def _cfs(x, cell, fto, period, device=torch.device('cpu'), type_complex=torch.complex128):
    """Exact Fourier coefficients of a piecewise-constant function, along the last axis.

    `x` holds the trailing edge of each segment and `cell` its value there. Differentiating a
    step function leaves a train of deltas at the edges, whose transform is a plain sum of
    phases; integrating puts back the 1 / (i 2 pi m) factor. So no quadrature is involved and
    the result is exact for the piecewise-constant object - the whole reason this path exists
    next to the FFT one.
    """
    # The jump at each edge: value of the next segment minus this one. Rolling makes the last
    # segment wrap onto the first, which is what periodicity means here.
    cell_next = torch.roll(cell, -1, dims=1)
    cell_diff = cell_next - cell
    cell_diff = cell_diff.type(type_complex)

    # Orders run to twice fto, not fto. The convolution matrix assembled downstream is indexed
    # by differences of two orders, and those reach +-2*fto.
    modes = torch.arange(-2 * fto, 2 * fto + 1, 1, device=device).type(type_complex)

    center = 2 * fto  # index of m = 0, since `modes` starts at -2*fto
    nc = torch.ones(len(modes), device=device).type(torch.bool)

    nc[center] = False

    # Segment widths. The last one closes on `period` rather than on a next edge.
    x_next = torch.vstack((torch.roll(x, -1, dims=0)[:-1], torch.tensor([period], device=device))) - x

    f = cell_diff @ torch.exp(-1j * 2 * torch.pi * x @ modes[None, :] / period).type(type_complex)

    f[:, nc] /= (1j * 2 * torch.pi * modes[nc])
    # m = 0 is excluded above because that division would be by zero. The DC coefficient is
    # just the width-weighted mean, which the deltas cannot express.
    f[:, center] = (cell @ torch.vstack((x[0], x_next[:-1]))).flatten() / period

    return f


def cfs2d(cell, x, y, conti_x, conti_y, fto_x, fto_y, device=torch.device('cpu'), type_complex=torch.complex128):
    """Convolution matrix from an exact (continuous) Fourier expansion.

    `x` and `y` are the segment edges, increasing, with the period as the last entry - the
    format `cell_compression` produces from a raster and the vector modeler produces directly.
    """
    cell = cell.type(type_complex)
    x = x.type(type_complex)
    y = y.type(type_complex)

    ff_x = 2 * fto_x + 1
    ff_y = 2 * fto_y + 1

    # The period is read off the edge list rather than passed in: the last edge *is* the
    # period, by the convention above.
    period_x, period_y = x[-1], y[-1]

    # _cfs works along the last axis, so transposing puts y there and makes the first pass the
    # y pass. The x pass below transposes back.
    cell = cell.T

    if conti_y == 0:
        cell = 1 / cell

    cfs1d = _cfs(y, cell, fto_y, period_y, device=device, type_complex=type_complex)

    # `circulant` is indexed from mode 0; _cfs returns a row starting at mode -2*fto, so the
    # offset re-centres the lookup. dfs2d below needs no offset - see there.
    conv_index_1 = circulant(fto_y, device=device) + (2 * fto_y)
    conv_index_2 = circulant(fto_x, device=device) + (2 * fto_x)

    conv1d = cfs1d[:, conv_index_1]

    # Exactly one direction on the inverse rule is the mixed case, and there Li's rule applies
    # to the inner blocks: build the 1D convolution matrix, then invert it, before the second
    # direction is transformed at all. With both directions on the same rule there is nothing
    # mixed and the blocks are used as they are.
    if conti_x ^ conti_y:
        conv1d = torch.linalg.inv(conv1d)

    conv1d = conv1d.reshape((-1, ff_y ** 2))

    cfs2d = _cfs(x, conv1d.T, fto_x, period_x, device=device, type_complex=type_complex)

    # Four indices (y_out, y_in, x_out, x_in) collapse to two. moveaxis pairs the two output
    # indices and the two input indices before the reshape, so the flat row index runs over
    # (y_out, x_out) and the column over (y_in, x_in) - the order the solver's field vectors
    # are stacked in.
    conv2d = cfs2d[:, conv_index_2]
    conv2d = conv2d.reshape((ff_y, ff_y, ff_x, ff_x))
    conv2d = torch.moveaxis(conv2d, 1, 2)
    conv2d = conv2d.reshape((ff_y*ff_x, ff_y*ff_x))

    # x on the inverse rule is applied last, to the assembled 2D matrix.
    if conti_x == 0:
        conv2d = torch.linalg.inv(conv2d)

    return conv2d


def dfs2d(cell, conti_x, conti_y, fto_x, fto_y, device=torch.device('cpu'), type_complex=torch.complex128):
    """Convolution matrix from an FFT over a uniform raster.

    Same structure as `cfs2d`, and the same factorization rules; only the transform differs.
    There are no coordinates here - the raster's own uniform spacing is the geometry, which is
    why a flip of the input array is a change of structure and not a change of convention.
    """
    cell = cell.type(type_complex)

    ff_x = 2 * fto_x + 1
    ff_y = 2 * fto_y + 1

    cell = cell.T

    if conti_y == 0:
        cell = 1 / cell

    # Dividing by the transform length is the DFT's 1/N: it makes the coefficients the Fourier
    # series' own, independent of how finely the layer happens to be sampled.
    dfs1d = torch.fft.fft(cell / cell.shape[1])

    # No offset, unlike cfs2d: fft returns modes in wraparound order, so index 0 is already
    # m = 0 and a negative index reaches the negative order directly.
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
    """Index matrix with circ[r, c] = c - r, for gathering coefficients into a Toeplitz block.

    A convolution in Fourier space is the matrix whose (r, c) entry is the coefficient of order
    c - r. Rather than build that matrix by arithmetic on values, this returns the *indices*
    and the caller gathers with them, so the values stay tensors on the autograd graph.

    Returns indices, not coefficients: negative entries are meant to be negative, and how they
    resolve depends on the layout of the row being gathered from (see the two call sites).
    """
    ff = 2 * fto + 1
    stride = 2 * fto
    circ = torch.zeros((ff, ff), device=device).type(torch.int)
    for r in range(stride + 1):
        idx = torch.arange(-r, -r + ff, 1, device=device)
        circ[r] = idx

    return circ
