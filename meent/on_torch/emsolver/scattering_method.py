"""Scattering-matrix (S-matrix) formulation of the layer stack.

Not reachable from the torch solver as it stands. `_base` imports these names but every
`solve_*` there goes through `transfer_method` instead, so nothing calls them; they are kept
as the reference implementation of the alternative formulation.

Two things follow from that and are worth knowing before using them. The code is numpy, not
torch - no device, no autograd, so a result cannot be differentiated or moved to a GPU. And
the arguments still use the older names (`n_I` / `n_II` for the superstrate and substrate
indices, `E_conv` for the permittivity convolution matrix), which no longer match the solver's.

The formulation itself is worth the file: S-matrices compose by the Redheffer star product,
which stays bounded for evanescent orders where the transfer matrix's growing exponentials
overflow. That is the trade - stability against the extra cost of the star products.

Functions are numbered in call order: `_1` sets up the half spaces, `_2` folds in one layer
and is called per layer, `_3` closes the stack and turns amplitudes into efficiencies.
"""

from .smm_util import *


def scattering_1d_1(k0, n_I, n_II, theta, phi, fourier_indices, period, pol, wl=None):
    """Half-space setup: the gap medium and both outer media, and the reflection-side S-matrix.

    The "gap" is a fictitious medium every layer is referenced to. Nothing physical sits there;
    referencing all layers to one common basis is what lets their S-matrices be composed
    without carrying each layer's own eigenvectors through the product.
    """

    kx_vector = (n_I * np.sin(theta) * np.cos(phi) - fourier_indices * (
                2 * np.pi / k0 / period[0])).astype('complex')
    Kx = np.diag(kx_vector)

    Wg, Vg, Kzg = homogeneous_1D(Kx, 1, wl=wl, comment='Gap')

    Wr, Vr, Kzr = homogeneous_1D(Kx, n_I, pol=pol, wl=wl, comment='Refl')

    Wt, Vt, Kzt = homogeneous_1D(Kx, n_II, pol=pol, wl=wl, comment='Tran')

    Ar, Br = A_B_matrices_half_space(Vr, Vg)
    _, Sg = S_RT(Ar, Br, ref_mode=True)

    return Kx, Wg, Vg, Kzg, Wr, Vr, Kzr, Wt, Vt, Kzt, Ar, Br, Sg


def scattering_1d_2(W, Wg, V, Vg, d, k0, LAMBDA, Sg):
    """Fold one layer into the running S-matrix `Sg`. Called once per layer, top to bottom."""
    A, B = A_B_matrices(W, Wg, V, Vg)
    _, S_dict = S_layer(A, B, d, k0, LAMBDA)
    _, Sg = RedhefferStar(Sg, S_dict)

    return A, B, S_dict, Sg


def scattering_1d_3(Wt, Wg, Vt, Vg, Sg, ff, Wr, fourier_order, Kzr, Kzt, n_I, n_II, theta, pol):
    At, Bt = A_B_matrices_half_space(Vt, Vg)
    _, St_dict = S_RT(At, Bt, ref_mode=False)
    _, Sg = RedhefferStar(Sg, St_dict)

    k_inc = n_I * np.array([np.sin(theta), 0, np.cos(theta)])

    # A unit amplitude in the zero order only - `fourier_order` is that order's index, the
    # middle of the range. Converting to the eigenvector basis is what `inv(Wr) @` does.
    c_inc = np.zeros((ff, 1))
    c_inc[fourier_order] = 1
    c_inc = np.linalg.inv(Wr) @ c_inc
    reflected = Wr @ Sg['S11'] @ c_inc
    transmitted = Wt @ Sg['S21'] @ c_inc

    rsq = np.square(np.abs(reflected))
    tsq = np.square(np.abs(transmitted))

    # Diffraction efficiency is a power ratio, so each order is weighted by its own real kz -
    # the part of it that actually carries power away - over the incident kz. Evanescent
    # orders have kz purely imaginary and drop out here, as they should.
    if pol == 0:
        de_ri = np.real(Kzr) @ rsq / np.real(k_inc[2])
        de_ti = np.real(Kzt) @ tsq / np.real(k_inc[2])
    elif pol == 1:
        # TM carries the index factors because the conserved quantity is built from H, not E,
        # and the two are related by the local index.
        de_ri = np.real(Kzr)@rsq/np.real(k_inc[2]) / n_I**2
        de_ti = np.real(Kzt)@tsq/np.real(k_inc[2]) * n_I**2 / n_II**4
    else:
        raise ValueError

    return de_ri.flatten(), de_ti.flatten()


def scattering_2d_1(n_I, n_II, theta, phi, k0, period, fourier_order):
    kx_inc = n_I * np.sin(theta) * np.cos(phi)
    ky_inc = n_I * np.sin(theta) * np.sin(phi)
    kz_inc = np.sqrt(n_I ** 2 * 1 - kx_inc ** 2 - ky_inc ** 2)

    Kx, Ky = K_matrix_cubic_2D(kx_inc, ky_inc, k0, period[0], period[1], fourier_order[0], fourier_order[1])

    e_h = 1
    Wg, Vg, Kzg = homogeneous_module(Kx, Ky, e_h)

    e_r = n_I ** 2
    Wr, Vr, Kzr = homogeneous_module(Kx, Ky, e_r)

    e_t = n_II ** 2
    Wt, Vt, Kzt = homogeneous_module(Kx, Ky, e_t)

    Ar, Br = A_B_matrices_half_space(Vr, Vg)

    _, Sr_dict = S_RT(Ar, Br, ref_mode=True)
    Sg = Sr_dict

    return Kx, Ky, kz_inc, Wg, Vg, Kzg, Wr, Vr, Kzr, Wt, Vt, Kzt, Ar, Br, Sg


def scattering_2d_2(W, Wg, V, Vg, d, k0, Sg, LAMBDA):

    A, B = A_B_matrices(W, Wg, V, Vg)
    _, Sl_dict = S_layer(A, B, d, k0, LAMBDA)
    Sg_matrix, Sg = RedhefferStar(Sg, Sl_dict)

    return A, B, Sl_dict, Sg_matrix, Sg


def scattering_2d_3(Wt, Wg, Vt, Vg, Sg, Wr, Kx, Ky, Kzr, Kzt, kz_inc, n_I, pol, theta,
                    phi, fourier_order, ff):
    normal_vector = np.array([0, 0, 1])

    if pol == 0:
        pte = 1
        ptm = 0
    elif pol == 1:
        pte = 0
        ptm = 1
    else:
        raise ValueError

    M = N = fourier_order
    NM = ff ** 2

    At, Bt = A_B_matrices_half_space(Vt, Vg)
    _, ST_dict = S_RT(At, Bt, ref_mode=False)

    Sg_matrix, Sg = RedhefferStar(Sg, ST_dict)


    K_inc_vector = n_I * np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])

    _, e_src, _ = initial_conditions(K_inc_vector, theta, normal_vector, pte, ptm, N, M)

    c_inc = np.linalg.inv(Wr) @ e_src
    reflected = Wr @ Sg['S11'] @ c_inc
    transmitted = Wt @ Sg['S21'] @ c_inc

    # The solved vector stacks x over y; z is not solved for.
    rx = reflected[0:NM, :]
    ry = reflected[NM:, :]
    tx = transmitted[0:NM, :]
    ty = transmitted[NM:, :]

    # z comes from the divergence condition instead: k . E = 0 in a homogeneous medium, so
    # Ez is fixed once Ex and Ey are known. Solving for it would add nothing.
    rz = np.linalg.inv(Kzr) @ (Kx @ rx + Ky @ ry)
    tz = np.linalg.inv(Kzt) @ (Kx @ tx + Ky @ ty)

    rsq = np.square(np.abs(rx)) + np.square(np.abs(ry)) + np.square(np.abs(rz))
    tsq = np.square(np.abs(tx)) + np.square(np.abs(ty)) + np.square(np.abs(tz))

    de_ri = np.real(Kzr)@rsq/np.real(K_inc_vector[2])
    de_ti = np.real(Kzt)@tsq/np.real(K_inc_vector[2])

    return de_ri, de_ti


def scattering_2d_wv(ff, Kx, Ky, E_conv, oneover_E_conv, oneover_E_conv_i, E_i, mu_conv=None):
    """Eigenmodes of one patterned layer: W (fields), V (their partners), LAMBDA (decay rates).

    Non-magnetic unless told otherwise, hence the identity default for mu.
    """
    NM = ff ** 2
    if mu_conv is None:
        mu_conv = np.identity(NM)

    P, Q, _ = P_Q_kz(Kx, Ky, E_conv, mu_conv, oneover_E_conv, oneover_E_conv_i, E_i)
    GAMMA = P @ Q

    # Maxwell's curl equations become a first-order pair in z; eliminating one field gives this
    # second-order system, whose eigenvalues are therefore kz^2. The square root is the step
    # back to kz, and it is taken on the diagonal because that is where the eigenvalues live.
    Lambda, W = np.linalg.eig(GAMMA)
    LAMBDA = np.diag(Lambda)
    LAMBDA = np.sqrt(LAMBDA.astype('complex'))

    # V is not a second eigenvector set: it is the partner field of W, obtained by applying the
    # half of the system that was eliminated and undoing the kz scaling.
    V = Q @ W @ np.linalg.inv(LAMBDA)

    return W, V, LAMBDA
