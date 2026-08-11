import torch

from .primitives import Eig, meeinv


def branch_sign(value):
    """+1 or -1 from the sign of Re(value), with sign(0) folded to +1.

    Used to orient each diffracted order's local transverse (s/p) basis, which reverses
    when an order crosses from +kx to -kx. The factor carries orientation only, so it must
    be exactly +-1. ``torch.sign`` is wrong for that: it returns 0 at 0, which multiplies
    an order's amplitude by zero instead of flipping its sign. kx_m = 0 is not a
    degenerate order - it is the one diffracted straight along z, where kz is largest of
    all - so torch.sign silently deleted the most strongly propagating order whenever
    n_top*sin(theta)*cos(phi) landed exactly on -m*wavelength/period, and R + T stopped
    summing to one. Every other input is unaffected: away from exactly 0 this reproduces
    torch.sign bit for bit.
    """
    ones = torch.ones_like(value.real)
    return torch.where(value.real < 0, -ones, ones)


def transfer_1d_1(pol, kx, n_top, n_bot, device=torch.device('cpu'), type_complex=torch.complex128):
    ff_x = len(kx)

    kz_top = (n_top ** 2 - kx ** 2) ** 0.5
    kz_bot = (n_bot ** 2 - kx ** 2) ** 0.5

    # For evanescent orders the sqrt argument sits on (or next to) the negative real
    # axis, where the branch is ambiguous and the returned root can be the growing one.
    # Force the decaying branch there. Compare against the real part of the index: the
    # propagating/evanescent cutoff is only defined for a lossless medium, and a complex
    # index cannot be ordered against a real magnitude at all.
    evan_top = abs(kx) > abs(torch.as_tensor(n_top).real)
    evan_bot = abs(kx) > abs(torch.as_tensor(n_bot).real)
    kz_top[evan_top] = -1j * (kx[evan_top] ** 2 - n_top ** 2) ** 0.5
    kz_bot[evan_bot] = -1j * (kx[evan_bot] ** 2 - n_bot ** 2) ** 0.5

    F = torch.eye(ff_x, device=device, dtype=type_complex)

    if pol == 0:  # TE
        Kz_bot = torch.diag(kz_bot)
        G = 1j * Kz_bot
    elif pol == 1:  # TM
        Kz_bot = torch.diag(kz_bot / (n_bot ** 2))
        G = 1j * Kz_bot
    else:
        raise ValueError

    T = torch.eye(ff_x, device=device, dtype=type_complex)

    return kz_top, kz_bot, F, G, T


def transfer_1d_2(pol, kx, epx_conv, epy_conv, epz_conv_i, device=torch.device('cpu'), type_complex=torch.complex128,
                  perturbation=1E-10, use_pinv=False):

    Kx = torch.diag(kx)

    if pol == 0:
        A = Kx ** 2 - epy_conv

        Eig.perturbation = perturbation
        eigenvalues, W = Eig.apply(A)

        # eigenvalues += 0j  # to get positive square root
        q = eigenvalues ** 0.5
        Q = torch.diag(q)
        V = W @ Q

    elif pol == 1:
        B = Kx @ epz_conv_i @ Kx - torch.eye(epy_conv.shape[0], device=device, dtype=type_complex)

        Eig.perturbation = perturbation
        eigenvalues, W = Eig.apply(epx_conv @ B)

        # eigenvalues += 0j  # to get positive square root
        q = eigenvalues ** 0.5

        Q = torch.diag(q)
        V = meeinv(epx_conv, use_pinv) @ W @ Q

    else:
        raise ValueError

    return W, V, q


def transfer_1d_3(k0, W, V, q, d, F, G, T, device=torch.device('cpu'), type_complex=torch.complex128, use_pinv=False):
    ff_x = len(q)

    I = torch.eye(ff_x, device=device, dtype=type_complex)

    X = torch.diag(torch.exp(-k0 * q * d))

    W_i = meeinv(W, use_pinv)
    V_i = meeinv(V, use_pinv)

    A = 0.5 * (W_i @ F + V_i @ G)
    B = 0.5 * (W_i @ F - V_i @ G)

    A_i = meeinv(A, use_pinv)

    F = W @ (I + X @ B @ A_i @ X)
    G = V @ (I - X @ B @ A_i @ X)
    T = T @ A_i @ X

    return X, F, G, T, A_i, B


def transfer_1d_4(pol, ff_x, F, G, T, kx, kz_top, kz_bot, theta, n_top, n_bot, device=torch.device('cpu'),
                  type_complex=torch.complex128, use_pinv=False):

    Kz_top = torch.diag(kz_top)
    kz_top = kz_top.reshape((1, ff_x))
    kz_bot = kz_bot.reshape((1, ff_x))

    delta_i0 = torch.zeros(ff_x, device=device, dtype=type_complex)
    delta_i0[ff_x // 2] = 1

    if pol == 0:  # TE
        inc_term = 1j * n_top * torch.cos(theta) * delta_i0
        T1 = meeinv(G + 1j * Kz_top @ F, use_pinv) @ (1j * Kz_top @ delta_i0 + inc_term)

    elif pol == 1:  # TM
        inc_term = 1j * delta_i0 * torch.cos(theta) / n_top
        T1 = meeinv(G + 1j * Kz_top / (n_top ** 2) @ F, use_pinv) @ (1j * Kz_top / (n_top ** 2) @ delta_i0 + inc_term)
    else:
        raise ValueError

    # T1 = np.linalg.pinv(G + 1j * YZ_I @ F) @ (1j * YZ_I @ delta_i0 + inc_term)
    # Conjugated to put this path on the same time convention as the conical and 2D solvers.
    # Without it a 1D grating at phi = 0 and the same grating at phi = 1e-7 -- physically the
    # same problem, but routed to different solvers -- returned coefficients that differed by
    # a complex conjugate. The efficiencies were identical either way, so nothing caught it.
    # R_s/R_p/T_s/T_p are public order-local modal amplitudes, not fixed-Cartesian field
    # components.  The local transverse basis reverses when an outgoing order crosses from
    # +kx to -kx.  Use the incident branch as the reference, exactly as the conical/2D
    # paths do, so the pure-1D solver exposes the same coefficient definition.
    #
    # See branch_sign for why torch.sign cannot be used here. The incident order sits at
    # index ff_x // 2, matching delta_i0 above; reading its kx directly rather than
    # sign(theta) also drops the old dependence on theta being perturbed away from zero.
    phase_conv = branch_sign(kx) * branch_sign(kx[ff_x // 2])
    R = (F @ T1 - delta_i0).reshape((1, ff_x)).conj() * phase_conv
    T = (T @ T1).reshape((1, ff_x)).conj() * phase_conv

    # de_ri = np.real(np.real(R * np.conj(R) * kz_top / (n_top * np.cos(theta))))
    # de_ri = np.real(R * np.conj(R) * np.real(kz_top / (n_top * np.cos(theta))))
    de_ri = (R * R.conj() * (kz_top / (n_top * torch.cos(theta))).real).real

    if pol == 0:
        # de_ti = np.real(T * np.conj(T) * np.real(kz_bot / (n_top * np.cos(theta))))
        # de_ti = np.real(T * np.conj(T) * np.real(kz_bot / (n_top * np.cos(theta))))
        de_ti = (T * T.conj() * (kz_bot / (n_top * torch.cos(theta))).real).real
        R_s = R
        R_p = torch.zeros_like(R)
        T_s = T
        T_p = torch.zeros_like(T)
        de_ri_s = de_ri
        de_ri_p = torch.zeros_like(de_ri)
        de_ti_s = de_ti
        de_ti_p = torch.zeros_like(de_ti)

    elif pol == 1:
        # de_ti = np.real(T * np.conj(T) * np.real(kz_bot / n_bot ** 2) / (np.cos(theta) / n_top))
        # de_ti = np.real(T * np.conj(T) * np.real(kz_bot / n_bot ** 2 / (np.cos(theta) / n_top)))
        de_ti = (T * T.conj() * (kz_bot / n_bot ** 2 / (torch.cos(theta) / n_top)).real).real
        R_s = torch.zeros_like(R)
        R_p = R
        T_s = torch.zeros_like(T)
        T_p = T
        de_ri_s = torch.zeros_like(de_ri)
        de_ri_p = de_ri
        de_ti_s = torch.zeros_like(de_ti)
        de_ti_p = de_ti
    else:
        raise ValueError

    res = {'R_s': R_s, 'R_p': R_p, 'T_s': T_s, 'T_p': T_p,
           'de_ri': de_ri, 'de_ri_s': de_ri_s, 'de_ri_p': de_ri_p,
           'de_ti': de_ti, 'de_ti_s': de_ti_s, 'de_ti_p': de_ti_p,
           }

    result = {'res': res}

    return result, T1


def transfer_1d_conical_1(kx, ky, n_top, n_bot, device='cpu', type_complex=torch.complex128):
    ff_x = len(kx)
    ff_y = len(ky)
    ff_xy = ff_x * ff_y

    I = torch.eye(ff_xy, device=device, dtype=type_complex)
    O = torch.zeros((ff_xy, ff_xy), device=device, dtype=type_complex)

    # TODO: cleaning
    # ky = k0 * n_I * torch.sin(theta) * torch.sin(phi)
    #
    # k_I_z = (k0 ** 2 * n_I ** 2 - kx_vector ** 2 - ky ** 2) ** 0.5
    # k_II_z = (k0 ** 2 * n_II ** 2 - kx_vector ** 2 - ky ** 2) ** 0.5
    #
    # k_I_z = torch.conj(k_I_z.flatten())
    # k_II_z = torch.conj(k_II_z.flatten())
    #
    # Kx = torch.diag(kx_vector / k0)


    kz_top = (n_top ** 2 - kx ** 2 - ky.reshape((-1, 1)) ** 2) ** 0.5
    kz_bot = (n_bot ** 2 - kx ** 2 - ky.reshape((-1, 1)) ** 2) ** 0.5

    # Force the decaying branch of the sqrt for evanescent orders; see transfer_1d_1.
    k_par2 = kx ** 2 + ky.reshape((-1, 1)) ** 2
    evan_top = abs(k_par2) > abs(torch.as_tensor(n_top).real ** 2)
    evan_bot = abs(k_par2) > abs(torch.as_tensor(n_bot).real ** 2)
    kz_top[evan_top] = -1j * (k_par2[evan_top] - n_top ** 2) ** 0.5
    kz_bot[evan_bot] = -1j * (k_par2[evan_bot] - n_bot ** 2) ** 0.5

    kz_top = kz_top.flatten()
    kz_bot = kz_bot.flatten()


    # varphi = torch.arctan(ky / kx_vector)

    varphi = torch.arctan(ky.reshape((-1, 1)) / kx).flatten()
    Kz_bot = torch.diag(kz_bot)


    # Y_I = torch.diag(k_I_z / k0)
    # Y_II = torch.diag(k_II_z / k0)
    #
    # Z_I = torch.diag(k_I_z / (k0 * n_I ** 2))
    # Z_II = torch.diag(k_II_z / (k0 * n_II ** 2))

    big_F = torch.cat(
        [
            torch.cat([I, O], dim=1),
            torch.cat([O, 1j * Kz_bot / (n_bot ** 2)], dim=1),
        ]
    )

    big_G = torch.cat(
        [
            torch.cat([1j * Kz_bot, O], dim=1),
            torch.cat([O, I], dim=1),
        ]
    )

    big_T = torch.eye(2*ff_xy, device=device, dtype=type_complex)
    return kz_top, kz_bot, varphi, big_F, big_G, big_T

    # return Kx, ky, k_I_z, k_II_z, varphi, Y_I, Y_II, Z_I, Z_II, big_F, big_G, big_T


# def transfer_1d_conical_2(k0, Kx, ky, E_conv, E_i, o_E_conv_i, ff, d, varphi, big_F, big_G, big_T,
#                           device='cpu', type_complex=torch.complex128, perturbation=1E-10):
def transfer_1d_conical_2(kx, ky, epx_conv, epy_conv, epz_conv_i, device='cpu', type_complex=torch.complex128,
                          perturbation=1E-10, use_pinv=False):

    ff_x = len(kx)
    ff_y = len(ky)
    ff_xy = ff_x * ff_y

    I = torch.eye(ff_xy, device=device, dtype=type_complex)

    Kx = torch.diag(kx.tile(ff_y).flatten())
    Ky = torch.diag(ky.reshape((-1, 1)).tile(ff_x).flatten())

    A = Kx ** 2 - epy_conv
    B = Kx @ epz_conv_i @ Kx - I

    Omega2_RL = Ky ** 2 + A
    Omega2_LR = Ky ** 2 + B @ epx_conv

    Eig.perturbation = perturbation
    eigenvalues_1, W_1 = Eig.apply(Omega2_RL)
    eigenvalues_2, W_2 = Eig.apply(Omega2_LR)

    q_1 = eigenvalues_1 ** 0.5
    q_2 = eigenvalues_2 ** 0.5

    Q_1 = torch.diag(q_1)
    Q_2 = torch.diag(q_2)

    A_i = meeinv(A, use_pinv)
    B_i = meeinv(B, use_pinv)

    V_11 = A_i @ W_1 @ Q_1
    V_12 = Ky @ A_i @ Kx @ W_2
    V_21 = Ky @ B_i @ Kx @ epz_conv_i @ W_1
    V_22 = B_i @ W_2 @ Q_2

    W = torch.cat([W_1, W_2], dim=1)
    V = torch.cat(
        [
            torch.cat([V_11, V_12], dim=1),
            torch.cat([V_21, V_22], dim=1),
        ])

    q = torch.hstack([q_1, q_2])

    return W, V, q


# def transfer_1d_conical_3(big_F, big_G, big_T, Z_I, Y_I, psi, theta, ff, delta_i0, k_I_z, k0, n_I, n_II, k_II_z,
#                           device='cpu', type_complex=torch.complex128):
def transfer_1d_conical_3(k0, W, V, q, d, varphi, big_F, big_G, big_T, device='cpu', type_complex=torch.complex128,
                          use_pinv=False):

    ff_xy = len(q) // 2
    I = torch.eye(ff_xy, device=device, dtype=type_complex)
    O = torch.zeros((ff_xy, ff_xy), device=device, dtype=type_complex)

    q_1 = q[:ff_xy]
    q_2 = q[ff_xy:]

    W_1 = W[:, :ff_xy]
    W_2 = W[:, ff_xy:]

    V_11 = V[:ff_xy, :ff_xy]
    V_12 = V[:ff_xy, ff_xy:]
    V_21 = V[ff_xy:, :ff_xy]
    V_22 = V[ff_xy:, ff_xy:]


    X_1 = torch.diag(torch.exp(-k0 * q_1 * d))
    X_2 = torch.diag(torch.exp(-k0 * q_2 * d))

    F_c = torch.diag(torch.cos(varphi))
    F_s = torch.diag(torch.sin(varphi))

    V_ss = F_c @ V_11
    V_sp = F_c @ V_12 - F_s @ W_2
    W_ss = F_c @ W_1 + F_s @ V_21
    W_sp = F_s @ V_22
    V_ps = F_s @ V_11
    V_pp = F_c @ W_2 + F_s @ V_12
    W_ps = F_c @ V_21 - F_s @ W_1
    W_pp = F_c @ V_22

    big_I = torch.eye(2 * (len(I)), device=device, dtype=type_complex)

    big_X = torch.cat([
        torch.cat([X_1, O], dim=1),
        torch.cat([O, X_2], dim=1)])

    big_W = torch.cat([
        torch.cat([V_ss, V_sp], dim=1),
        torch.cat([V_ps, V_pp], dim=1)])

    big_V = torch.cat([
        torch.cat([W_ss, W_sp],  dim=1),
        torch.cat([W_ps, W_pp], dim=1)])

    big_W_i = meeinv(big_W, use_pinv)
    big_V_i = meeinv(big_V, use_pinv)

    big_A = 0.5 * (big_W_i @ big_F + big_V_i @ big_G)
    big_B = 0.5 * (big_W_i @ big_F - big_V_i @ big_G)

    big_A_i = meeinv(big_A, use_pinv)

    big_F = big_W @ (big_I + big_X @ big_B @ big_A_i @ big_X)
    big_G = big_V @ (big_I - big_X @ big_B @ big_A_i @ big_X)

    big_T = big_T @ big_A_i @ big_X

    return big_X, big_F, big_G, big_T, big_A_i, big_B


def transfer_1d_conical_4(kx, ff_x, ff_y, big_F, big_G, big_T, kz_top, kz_bot, psi, theta, n_top, n_bot, device='cpu',
                          type_complex=torch.complex128, use_pinv=False):

    ff_xy = ff_x * ff_y

    Kz_top = torch.diag(kz_top)
    kz_top = kz_top.reshape((ff_y, ff_x))
    kz_bot = kz_bot.reshape((ff_y, ff_x))

    I = torch.eye(ff_xy, device=device, dtype=type_complex)
    O = torch.zeros((ff_xy, ff_xy), device=device, dtype=type_complex)

    big_F_11 = big_F[:ff_xy, :ff_xy]
    big_F_12 = big_F[:ff_xy, ff_xy:]
    big_F_21 = big_F[ff_xy:, :ff_xy]
    big_F_22 = big_F[ff_xy:, ff_xy:]

    big_G_11 = big_G[:ff_xy, :ff_xy]
    big_G_12 = big_G[:ff_xy, ff_xy:]
    big_G_21 = big_G[ff_xy:, :ff_xy]
    big_G_22 = big_G[ff_xy:, ff_xy:]

    delta_i0 = torch.zeros((ff_xy, 1), device=device, dtype=type_complex)
    delta_i0[ff_xy // 2, 0] = 1

    # Final Equation in form of AX=B
    final_A = torch.cat(
        [
            torch.cat([I, O, -big_F_11, -big_F_12], dim=1),
            torch.cat([O, -1j * Kz_top / (n_top ** 2), -big_F_21, -big_F_22], dim=1),
            torch.cat([-1j * Kz_top, O, -big_G_11, -big_G_12], dim=1),
            torch.cat([O, I, -big_G_21, -big_G_22], dim=1),
        ]
    )

    final_B = torch.cat(
        [
            torch.cat([-torch.sin(psi) * delta_i0], dim=1),
            torch.cat([-torch.cos(psi) * torch.cos(theta) * delta_i0], dim=1),
            torch.cat([-1j * torch.sin(psi) * n_top * torch.cos(theta) * delta_i0], dim=1),
            torch.cat([1j * n_top * torch.cos(psi) * delta_i0], dim=1),
        ]
    )

    final_A_inv = meeinv(final_A, use_pinv)
    final_RT = final_A_inv @ final_B

    # Reflected and transmitted amplitudes have to carry the SAME phase convention.
    # The conjugation and the sign(kx) branch correction used to sit on R only, which left
    # R on the analytic (Fresnel) convention and T on its conjugate. That is invisible in the
    # efficiencies -- |conj(z)| = |z|, and sign(kx) is +-1 -- so every efficiency-based check
    # passed while the coefficients disagreed with RETICOLO and with the Fresnel solution of a
    # plain slab. Anything that consumes R and T together (near field, S-matrix, coherent
    # stacking) was reading one of them with a flipped imaginary part.
    # See branch_sign: torch.sign returns 0 at 0, which would delete the order diffracted
    # straight along z instead of orienting it. The incident order sits at index ff_x // 2.
    phase_conv = branch_sign(kx) * branch_sign(kx[ff_x // 2])

    R_s = (final_RT[:ff_xy, :].reshape((ff_y, ff_x))).conj() * phase_conv
    R_p = ((1j/n_top)*final_RT[ff_xy: 2 * ff_xy, :].reshape((ff_y, ff_x))).conj() * phase_conv

    big_T1 = final_RT[2 * ff_xy:, :]
    # big_T_tetm = big_T.clone().detach()
    big_T_tetm = big_T.clone()
    big_T = big_T @ big_T1

    T_s = (big_T[:ff_xy, :].reshape((ff_y, ff_x))).conj() * phase_conv
    T_p = ((1j/n_bot)*big_T[ff_xy:, :].reshape((ff_y, ff_x))).conj() * phase_conv

    de_ri_s = (R_s * R_s.conj() * (kz_top / (n_top * torch.cos(theta))).real).real
    de_ri_p = (R_p * R_p.conj() * (kz_top / (n_top * torch.cos(theta))).real).real

    de_ti_s = (T_s * T_s.conj() * (kz_bot / (n_top * torch.cos(theta))).real).real
    de_ti_p = (T_p * T_p.conj() * (kz_bot / (n_top * torch.cos(theta))).real).real

    de_ri = de_ri_s + de_ri_p
    de_ti = de_ti_s + de_ti_p

    res = {'R_s': R_s, 'R_p': R_p, 'T_s': T_s, 'T_p': T_p,
           'de_ri_s': de_ri_s, 'de_ri_p': de_ri_p, 'de_ri': de_ri,
           'de_ti_s': de_ti_s, 'de_ti_p': de_ti_p, 'de_ti': de_ti}

    # TE TM incidence
    psi_tm = torch.tensor(0, dtype=type_complex, device=device)
    final_B_tm = torch.cat(
        [
            torch.cat([-torch.sin(psi_tm) * delta_i0], dim=1),
            torch.cat([-torch.cos(psi_tm) * torch.cos(theta) * delta_i0], dim=1),
            torch.cat([-1j * torch.sin(psi_tm) * n_top * torch.cos(theta) * delta_i0], dim=1),
            torch.cat([1j * n_top * torch.cos(psi_tm) * delta_i0], dim=1),
        ]
    )

    psi_te = torch.tensor(torch.pi / 2, dtype=type_complex, device=device)
    final_B_te = torch.cat(
        [
            torch.cat([-torch.sin(psi_te) * delta_i0], dim=1),
            torch.cat([-torch.cos(psi_te) * torch.cos(theta) * delta_i0], dim=1),
            torch.cat([-1j * torch.sin(psi_te) * n_top * torch.cos(theta) * delta_i0], dim=1),
            torch.cat([1j * n_top * torch.cos(psi_te) * delta_i0], dim=1),
        ]
    )

    final_B_tetm = torch.hstack([final_B_te, final_B_tm])
    final_RT_tetm = final_A_inv @ final_B_tetm

    R_s_tetm = (final_RT_tetm[:ff_xy, :].T.reshape((2, ff_y, ff_x))).conj() * phase_conv
    R_p_tetm = ((-1j/n_top)*final_RT_tetm[ff_xy: 2 * ff_xy, :].T.reshape((2, ff_y, ff_x))).conj() * phase_conv

    big_T1_tetm = final_RT_tetm[2 * ff_xy:, :]
    big_T_tetm = big_T_tetm @ big_T1_tetm

    T_s_tetm = big_T_tetm[:ff_xy, :].T.reshape((2, ff_y, ff_x))
    T_p_tetm = (-1j/n_bot)*big_T_tetm[ff_xy:, :].T.reshape((2, ff_y, ff_x))

    de_ri_s_tetm = (R_s_tetm * R_s_tetm.conj() * (kz_top / (n_top * torch.cos(theta))).real).real
    de_ri_p_tetm = (R_p_tetm * R_p_tetm.conj() * (kz_top / (n_top * torch.cos(theta))).real).real

    de_ti_s_tetm = (T_s_tetm * T_s_tetm.conj() * (kz_bot / (n_top * torch.cos(theta))).real).real
    de_ti_p_tetm = (T_p_tetm * T_p_tetm.conj() * (kz_bot / (n_top * torch.cos(theta))).real).real

    de_ri_tetm = de_ri_s_tetm + de_ri_p_tetm
    de_ti_tetm = de_ti_s_tetm + de_ti_p_tetm

    res_te_inc = {'R_s': R_s_tetm[0], 'R_p': R_p_tetm[0], 'T_s': T_s_tetm[0], 'T_p': T_p_tetm[0],
                  'de_ri_s': de_ri_s_tetm[0], 'de_ri_p': de_ri_p_tetm[0], 'de_ri': de_ri_tetm[0],
                  'de_ti_s': de_ti_s_tetm[0], 'de_ti_p': de_ti_p_tetm[0], 'de_ti': de_ti_tetm[0]}

    res_tm_inc = {'R_s': R_s_tetm[1], 'R_p': R_p_tetm[1], 'T_s': T_s_tetm[1], 'T_p': T_p_tetm[1],
                  'de_ri_s': de_ri_s_tetm[1], 'de_ri_p': de_ri_p_tetm[1], 'de_ri': de_ri_tetm[1],
                  'de_ti_s': de_ti_s_tetm[1], 'de_ti_p': de_ti_p_tetm[1], 'de_ti': de_ti_tetm[1]}

    result = {'res': res, 'res_tm_inc': res_tm_inc, 'res_te_inc': res_te_inc}
    big_T1_all = torch.stack((big_T1, big_T1_tetm[:, 0:1], big_T1_tetm[:, 1:2]))

    return result, big_T1_all


def transfer_2d_1(kx, ky, n_top, n_bot, device=torch.device('cpu'), type_complex=torch.complex128):
    ff_x = len(kx)
    ff_y = len(ky)
    ff_xy = ff_x * ff_y

    I = torch.eye(ff_xy, device=device, dtype=type_complex)
    O = torch.zeros((ff_xy, ff_xy), device=device, dtype=type_complex)

    kz_top = (n_top ** 2 - kx ** 2 - ky.reshape((-1, 1)) ** 2) ** 0.5
    kz_bot = (n_bot ** 2 - kx ** 2 - ky.reshape((-1, 1)) ** 2) ** 0.5

    # Force the decaying branch of the sqrt for evanescent orders; see transfer_1d_1.
    k_par2 = kx ** 2 + ky.reshape((-1, 1)) ** 2
    evan_top = abs(k_par2) > abs(torch.as_tensor(n_top).real ** 2)
    evan_bot = abs(k_par2) > abs(torch.as_tensor(n_bot).real ** 2)
    kz_top[evan_top] = -1j * (k_par2[evan_top] - n_top ** 2) ** 0.5
    kz_bot[evan_bot] = -1j * (k_par2[evan_bot] - n_bot ** 2) ** 0.5

    kz_top = kz_top.flatten()
    kz_bot = kz_bot.flatten()

    varphi = torch.arctan(ky.reshape((-1, 1)) / kx).flatten()
    Kz_bot = torch.diag(kz_bot)

    big_F = torch.cat(
        [
            torch.cat([I, O], dim=1),
            torch.cat([O, 1j * Kz_bot / (n_bot ** 2)], dim=1),
        ]
    )

    big_G = torch.cat(
        [
            torch.cat([1j * Kz_bot, O], dim=1),
            torch.cat([O, I], dim=1),
        ]
    )

    big_T = torch.eye(2 * ff_xy, device=device, dtype=type_complex)

    return kz_top, kz_bot, varphi, big_F, big_G, big_T


def transfer_2d_2(kx, ky, epx_conv, epy_conv, epz_conv_i, device=torch.device('cpu'), type_complex=torch.complex128,
                   perturbation=1E-10, use_pinv=False):

    ff_x = len(kx)
    ff_y = len(ky)
    ff_xy = ff_x * ff_y

    I = torch.eye(ff_xy, device=device, dtype=type_complex)

    Kx = torch.diag(kx.tile(ff_y).flatten())
    Ky = torch.diag(ky.reshape((-1, 1)).tile(ff_x).flatten())

    B = Kx @ epz_conv_i @ Kx - I
    D = Ky @ epz_conv_i @ Ky - I

    Omega2_LR = torch.cat(
        [
            torch.cat([Ky ** 2 + B @ epx_conv, Kx @ (epz_conv_i @ Ky @ epy_conv - Ky)], dim=1),
            torch.cat([Ky @ (epz_conv_i @ Kx @ epx_conv - Kx), Kx ** 2 + D @ epy_conv], dim=1)
        ])

    Eig.perturbation = perturbation
    eigenvalues, W = Eig.apply(Omega2_LR)
    q = eigenvalues ** 0.5

    q = torch.where(q.abs() < 1E-10, torch.full_like(q, 1E-10), q)

    Q = torch.diag(q)
    Q_i = torch.diag(1 / q)
    Omega_R = torch.cat(
        [
            torch.cat([-Kx @ Ky, Kx ** 2 - epy_conv], dim=1),
            torch.cat([epx_conv - Ky ** 2, Ky @ Kx], dim=1)
        ]
    )
    V = Omega_R @ W @ Q_i

    return W, V, q


def transfer_2d_3(k0, W, V, q, d, varphi, big_F, big_G, big_T, device=torch.device('cpu'),
                  type_complex=torch.complex128, use_pinv=False):
    ff_xy = len(q)//2

    I = torch.eye(ff_xy, device=device, dtype=type_complex)
    O = torch.zeros((ff_xy, ff_xy), device=device, dtype=type_complex)

    q_1 = q[:ff_xy]
    q_2 = q[ff_xy:]

    W_11 = W[:ff_xy, :ff_xy]
    W_12 = W[:ff_xy, ff_xy:]
    W_21 = W[ff_xy:, :ff_xy]
    W_22 = W[ff_xy:, ff_xy:]

    V_11 = V[:ff_xy, :ff_xy]
    V_12 = V[:ff_xy, ff_xy:]
    V_21 = V[ff_xy:, :ff_xy]
    V_22 = V[ff_xy:, ff_xy:]

    X_1 = torch.diag(torch.exp(-k0 * q_1 * d))
    X_2 = torch.diag(torch.exp(-k0 * q_2 * d))

    F_c = torch.diag(torch.cos(varphi))
    F_s = torch.diag(torch.sin(varphi))

    W_ss = F_c @ W_21 - F_s @ W_11
    W_sp = F_c @ W_22 - F_s @ W_12
    W_ps = F_c @ W_11 + F_s @ W_21
    W_pp = F_c @ W_12 + F_s @ W_22

    V_ss = F_c @ V_11 + F_s @ V_21
    V_sp = F_c @ V_12 + F_s @ V_22
    V_ps = F_c @ V_21 - F_s @ V_11
    V_pp = F_c @ V_22 - F_s @ V_12

    big_I = torch.eye(2 * (len(I)), device=device, dtype=type_complex)

    big_X = torch.cat([
        torch.cat([X_1, O], dim=1),
        torch.cat([O, X_2], dim=1)])

    big_W = torch.cat([
        torch.cat([W_ss, W_sp], dim=1),
        torch.cat([W_ps, W_pp], dim=1)])

    big_V = torch.cat([
        torch.cat([V_ss, V_sp],  dim=1),
        torch.cat([V_ps, V_pp], dim=1)])

    big_W_i = meeinv(big_W, use_pinv)
    big_V_i = meeinv(big_V, use_pinv)

    big_A = 0.5 * (big_W_i @ big_F + big_V_i @ big_G)
    big_B = 0.5 * (big_W_i @ big_F - big_V_i @ big_G)

    big_A_i = meeinv(big_A, use_pinv)

    big_F = big_W @ (big_I + big_X @ big_B @ big_A_i @ big_X)
    big_G = big_V @ (big_I - big_X @ big_B @ big_A_i @ big_X)

    big_T = big_T @ big_A_i @ big_X

    return big_X, big_F, big_G, big_T, big_A_i, big_B


def transfer_2d_4(kx, ff_x, ff_y, big_F, big_G, big_T, kz_top, kz_bot, psi, theta, n_top, n_bot,
                  device=torch.device('cpu'), type_complex=torch.complex128, use_pinv=False):

    ff_xy = ff_x * ff_y

    Kz_top = torch.diag(kz_top)
    kz_top = kz_top.reshape((ff_y, ff_x))
    kz_bot = kz_bot.reshape((ff_y, ff_x))

    I = torch.eye(ff_xy, device=device, dtype=type_complex)
    O = torch.zeros((ff_xy, ff_xy), device=device, dtype=type_complex)

    big_F_11 = big_F[:ff_xy, :ff_xy]
    big_F_12 = big_F[:ff_xy, ff_xy:]
    big_F_21 = big_F[ff_xy:, :ff_xy]
    big_F_22 = big_F[ff_xy:, ff_xy:]

    big_G_11 = big_G[:ff_xy, :ff_xy]
    big_G_12 = big_G[:ff_xy, ff_xy:]
    big_G_21 = big_G[ff_xy:, :ff_xy]
    big_G_22 = big_G[ff_xy:, ff_xy:]

    delta_i0 = torch.zeros((ff_xy, 1), device=device, dtype=type_complex)
    delta_i0[ff_xy // 2, 0] = 1

    # Final Equation in form of AX=B
    final_A = torch.cat(
        [
            torch.cat([I, O, -big_F_11, -big_F_12], dim=1),
            torch.cat([O, -1j * Kz_top / (n_top ** 2), -big_F_21, -big_F_22], dim=1),
            torch.cat([-1j * Kz_top, O, -big_G_11, -big_G_12], dim=1),
            torch.cat([O, I, -big_G_21, -big_G_22], dim=1),
        ]
    )

    final_B = torch.cat(
        [
            torch.cat([-torch.sin(psi) * delta_i0], dim=1),
            torch.cat([-torch.cos(psi) * torch.cos(theta) * delta_i0], dim=1),
            torch.cat([-1j * torch.sin(psi) * n_top * torch.cos(theta) * delta_i0], dim=1),
            torch.cat([1j * n_top * torch.cos(psi) * delta_i0], dim=1),
        ]
    )

    final_A_inv = meeinv(final_A, use_pinv)
    final_RT = final_A_inv @ final_B

    # Same convention fix as transfer_1d_conical_4 -- see the comment there.
    # See branch_sign: torch.sign returns 0 at 0, which would delete the order diffracted
    # straight along z instead of orienting it. The incident order sits at index ff_x // 2.
    phase_conv = branch_sign(kx) * branch_sign(kx[ff_x // 2])

    R_s = (final_RT[:ff_xy, :].reshape((ff_y, ff_x))).conj() * phase_conv
    R_p = ((1j/n_top)*final_RT[ff_xy: 2 * ff_xy, :].reshape((ff_y, ff_x))).conj() * phase_conv

    big_T1 = final_RT[2 * ff_xy:, :]
    # big_T_tetm = big_T.clone().detach()
    big_T_tetm = big_T.clone()
    big_T = big_T @ big_T1

    T_s = (big_T[:ff_xy, :].reshape((ff_y, ff_x))).conj() * phase_conv
    T_p = ((1j/n_bot)*big_T[ff_xy:, :].reshape((ff_y, ff_x))).conj() * phase_conv

    de_ri_s = (R_s * R_s.conj() * (kz_top / (n_top * torch.cos(theta))).real).real
    de_ri_p = (R_p * R_p.conj() * (kz_top / (n_top * torch.cos(theta))).real).real

    de_ti_s = (T_s * T_s.conj() * (kz_bot / (n_top * torch.cos(theta))).real).real
    de_ti_p = (T_p * T_p.conj() * (kz_bot / (n_top * torch.cos(theta))).real).real

    de_ri = de_ri_s + de_ri_p
    de_ti = de_ti_s + de_ti_p

    res = {'R_s': R_s, 'R_p': R_p, 'T_s': T_s, 'T_p': T_p,
           'de_ri_s': de_ri_s, 'de_ri_p': de_ri_p, 'de_ri': de_ri,
           'de_ti_s': de_ti_s, 'de_ti_p': de_ti_p, 'de_ti': de_ti}

    # TE TM incidence
    psi_tm = torch.tensor(0, dtype=type_complex, device=device)
    final_B_tm = torch.cat(
        [
            torch.cat([-torch.sin(psi_tm) * delta_i0], dim=1),
            torch.cat([-torch.cos(psi_tm) * torch.cos(theta) * delta_i0], dim=1),
            torch.cat([-1j * torch.sin(psi_tm) * n_top * torch.cos(theta) * delta_i0], dim=1),
            torch.cat([1j * n_top * torch.cos(psi_tm) * delta_i0], dim=1),
        ]
    )

    psi_te = torch.tensor(torch.pi/2, dtype=type_complex, device=device)
    final_B_te = torch.cat(
        [
            torch.cat([-torch.sin(psi_te) * delta_i0], dim=1),
            torch.cat([-torch.cos(psi_te) * torch.cos(theta) * delta_i0], dim=1),
            torch.cat([-1j * torch.sin(psi_te) * n_top * torch.cos(theta) * delta_i0], dim=1),
            torch.cat([1j * n_top * torch.cos(psi_te) * delta_i0], dim=1),
        ]
    )

    final_B_tetm = torch.hstack([final_B_te, final_B_tm])
    final_RT_tetm = final_A_inv @ final_B_tetm

    R_s_tetm = (final_RT_tetm[:ff_xy, :].T.reshape((2, ff_y, ff_x))).conj() * phase_conv
    R_p_tetm = ((-1j/n_top)*final_RT_tetm[ff_xy: 2 * ff_xy, :].T.reshape((2, ff_y, ff_x))).conj() * phase_conv

    big_T1_tetm = final_RT_tetm[2 * ff_xy:, :]
    big_T_tetm = big_T_tetm @ big_T1_tetm

    T_s_tetm = big_T_tetm[:ff_xy, :].T.reshape((2, ff_y, ff_x))
    T_p_tetm = (-1j/n_bot)*big_T_tetm[ff_xy:, :].T.reshape((2, ff_y, ff_x))

    de_ri_s_tetm = (R_s_tetm * R_s_tetm.conj() * (kz_top / (n_top * torch.cos(theta))).real).real
    de_ri_p_tetm = (R_p_tetm * R_p_tetm.conj() * (kz_top / (n_top * torch.cos(theta))).real).real

    de_ti_s_tetm = (T_s_tetm * T_s_tetm.conj() * (kz_bot / (n_top * torch.cos(theta))).real).real
    de_ti_p_tetm = (T_p_tetm * T_p_tetm.conj() * (kz_bot / (n_top * torch.cos(theta))).real).real

    de_ri_tetm = de_ri_s_tetm + de_ri_p_tetm
    de_ti_tetm = de_ti_s_tetm + de_ti_p_tetm

    res_te_inc = {'R_s': R_s_tetm[0], 'R_p': R_p_tetm[0], 'T_s': T_s_tetm[0], 'T_p': T_p_tetm[0],
                  'de_ri_s': de_ri_s_tetm[0], 'de_ri_p': de_ri_p_tetm[0], 'de_ri': de_ri_tetm[0],
                  'de_ti_s': de_ti_s_tetm[0], 'de_ti_p': de_ti_p_tetm[0], 'de_ti': de_ti_tetm[0]}

    res_tm_inc = {'R_s': R_s_tetm[1], 'R_p': R_p_tetm[1], 'T_s': T_s_tetm[1], 'T_p': T_p_tetm[1],
                  'de_ri_s': de_ri_s_tetm[1], 'de_ri_p': de_ri_p_tetm[1], 'de_ri': de_ri_tetm[1],
                  'de_ti_s': de_ti_s_tetm[1], 'de_ti_p': de_ti_p_tetm[1], 'de_ti': de_ti_tetm[1]}

    result = {'res': res, 'res_tm_inc': res_tm_inc, 'res_te_inc': res_te_inc}
    # big_T1_all = [big_T1, big_T1_tetm[:, 0:1], big_T1_tetm[:, 1:2]]
    big_T1_all = torch.stack((big_T1, big_T1_tetm[:, 0:1], big_T1_tetm[:, 1:2]))

    return result, big_T1_all
