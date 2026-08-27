import numpy as np
from numpy.linalg import inv, pinv
from scipy.linalg import block_diag


def A_B_matrices_half_space(V_layer, Vg):

    I = np.eye(len(Vg))
    a = I + inv(Vg) @ V_layer
    b = I - inv(Vg) @ V_layer

    return a, b


def A_B_matrices(W_layer, Wg, V_layer, Vg):
    W_i = inv(W_layer)
    V_i = inv(V_layer)

    a = W_i @ Wg + V_i @ Vg
    b = W_i @ Wg - V_i @ Vg

    return a, b


def S_layer(A, B, d, k0, modes):
    X = np.diag(np.exp(-np.diag(modes)*d*k0))

    A_i = inv(A)
    term_i = inv(A - X @ B @ A_i @ X @ B)

    S11 = term_i @ (X @ B @ A_i @ X @ A - B)
    S12 = term_i @ X @ (A - B @ A_i @ B)
    S22 = S11
    S21 = S12

    S_dict = {'S11': S11, 'S22': S22,  'S12': S12,  'S21': S21}
    S = np.block([[S11, S12], [S21, S22]])
    return S, S_dict


def S_RT(A, B, ref_mode):

    A_i = inv(A)

    S11 = -A_i @ B
    S12 = 2 * A_i
    S21 = 0.5*(A - B @ A_i @ B)
    S22 = B @ A_i

    if ref_mode:
        S_dict = {'S11': S11, 'S22': S22,  'S12': S12,  'S21': S21}
        S = np.block([[S11, S12], [S21, S22]])
    else:
        S_dict = {'S11': S22, 'S22': S11,  'S12': S21,  'S21': S12}
        S = np.block([[S22, S21], [S12, S11]])
    return S, S_dict


def homogeneous_module(Kx, Ky, e_r, m_r=1, perturbation=1E-10, wl=None, comment=None):
    assert type(Kx) == np.ndarray, 'not np.array'
    assert type(Ky) == np.ndarray, 'not np.array'

    N = len(Kx)
    I = np.identity(N)

    P = (e_r**-1)*np.block([[Kx*Ky, e_r*m_r*I-Kx**2], [Ky**2-m_r*e_r*I, -Ky*Kx]])
    Q = (e_r/m_r)*P

    diag = np.diag(Q)
    idx = np.nonzero(diag == 0)[0]
    if len(idx):
        Q[idx, idx] = np.conj(perturbation)
        print(wl, comment, 'non-invertible Q: adding perturbation')

    W = np.eye(N*2)
    Kz2 = (m_r*e_r*I-Kx**2-Ky**2).astype('complex')

    Kz = np.sqrt(Kz2)
    Kz = np.conj(Kz)

    diag = np.diag(Kz)
    idx = np.nonzero(diag == 0)[0]
    if len(idx):
        Kz[idx, idx] = perturbation
        print(wl, comment, 'non-invertible Kz: adding perturbation')

    eigenvalues = block_diag(1j*Kz, 1j*Kz)
    V = Q @ np.linalg.inv(eigenvalues)


    return W, V, Kz


def homogeneous_1D(Kx, n_index, m_r=1, pol=None, perturbation=1E-10, wl=None, comment=None):
    e_r = n_index ** 2

    I = np.identity(len(Kx))

    W = I
    Q = (1 / m_r) * (e_r * m_r * I - Kx ** 2)

    diag = np.diag(Q)
    idx = np.nonzero(diag == 0)[0]
    if len(idx):
        Q[idx, idx] = np.conj(perturbation)
        print(wl, comment, 'non-invertible Q: adding perturbation')

    Kz = np.sqrt(m_r*e_r*I-Kx**2)
    Kz = np.conj(Kz)

    diag = np.diag(Kz)
    idx = np.nonzero(diag == 0)[0]
    if len(idx):
        Kz[idx, idx] = perturbation
        print(wl, comment, 'non-invertible Kz: adding perturbation')

    if pol:
        Kz = Kz * (n_index ** 2)

    eigenvalues = -1j*Kz
    V = Q @ np.linalg.inv(eigenvalues)

    return W, V, Kz


def K_matrix_cubic_2D(beta_x, beta_y, k0, a_x, a_y, N_p, N_q):
    k_x = beta_x - 2*np.pi*np.arange(-N_p, N_p+1)/(k0*a_x)
    k_y = beta_y - 2*np.pi*np.arange(-N_q, N_q+1)/(k0*a_y)

    kx, ky = np.meshgrid(k_x, k_y)
    Kx = np.diag(kx.flatten())
    Ky = np.diag(ky.flatten())

    return Kx, Ky


def P_Q_kz(Kx, Ky, e_conv, mu_conv, oneover_E_conv, oneover_E_conv_i, E_i):
    argument = e_conv - Kx ** 2 - Ky ** 2
    Kz = np.conj(np.sqrt(argument.astype('complex')))

    P = np.block([
        [Kx @ E_i @ Ky, -Kx @ E_i @ Kx + mu_conv],
        [Ky @ E_i @ Ky - mu_conv,  -Ky @ E_i @ Kx]
    ])

    Q = np.block([
        [Kx @ inv(mu_conv) @ Ky, -Kx @ inv(mu_conv) @ Kx + e_conv],
        [-oneover_E_conv_i + Ky @ inv(mu_conv) @ Ky, -Ky @ inv(mu_conv) @ Kx]
    ])

    return P, Q, Kz


def delta_vector(P, Q):
    fourier_grid = np.zeros((P,Q))
    fourier_grid[int(P/2), int(Q/2)] = 1
    vector = fourier_grid.flatten()
    return np.matrix(np.reshape(vector, (1,len(vector))))


def initial_conditions(K_inc_vector, theta, normal_vector, pte, ptm, P, Q):
    if theta != 0:
        ate_vector = np.cross(K_inc_vector, normal_vector)
        ate_vector = ate_vector / (np.linalg.norm(ate_vector))
    else:
        ate_vector = np.array([0, 1, 0])

    atm_vector = np.cross(ate_vector, K_inc_vector)
    atm_vector = atm_vector / (np.linalg.norm(atm_vector))

    polarization = pte * ate_vector + ptm * atm_vector
    E_inc = polarization
    delta = delta_vector(2*P+1, 2*Q+1)

    e_src = np.hstack((polarization[0]*delta, polarization[1]*delta))
    e_src = np.matrix(e_src).T

    return E_inc, e_src, polarization


def RedhefferStar(SA, SB):
    assert type(SA) == dict, 'not dict'
    assert type(SB) == dict, 'not dict'

    SA_11, SA_12, SA_21, SA_22 = SA['S11'], SA['S12'], SA['S21'], SA['S22']
    SB_11, SB_12, SB_21, SB_22 = SB['S11'], SB['S12'], SB['S21'], SB['S22']
    N = len(SA_11)

    I = np.eye(N)
    D_i = inv(I - SB_11 @ SA_22)
    F_i = inv(I - SA_22 @ SB_11)

    SAB_11 = SA_11 + SA_12 @ D_i @ SB_11 @ SA_21
    SAB_12 = SA_12 @ D_i @ SB_12
    SAB_21 = SB_21 @ F_i @ SA_21
    SAB_22 = SB_22 + SB_21 @ F_i @ SA_22 @ SB_12

    SAB = np.block([[SAB_11, SAB_12], [SAB_21, SAB_22]])
    SAB_dict = {'S11': SAB_11, 'S22': SAB_22, 'S12': SAB_12, 'S21': SAB_21}

    return SAB, SAB_dict


def construct_global_scatter(scatter_list):
    Sr = scatter_list[0]
    Sg = Sr
    for i in range(1, len(scatter_list)):
        Sg = RedhefferStar(Sg, scatter_list[i])
    return Sg
