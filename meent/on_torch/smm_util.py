"""
currently SMM is not supported
"""
# many codes for scattering matrix method are from here:
# https://github.com/zhaonat/Rigorous-Coupled-Wave-Analysis
# also refer our fork https://github.com/yonghakim/zhaonat-rcwa

import numpy as np
from numpy.linalg import inv, pinv
# CHECK: try pseudo-inverse?
from scipy.linalg import block_diag
# CHECK: use numpy


def A_B_matrices_half_space(V_layer, Vg):

    I = np.eye(len(Vg))
    a = I + inv(Vg) @ V_layer
    b = I - inv(Vg) @ V_layer

    return a, b


def A_B_matrices(W_layer, Wg, V_layer, Vg):
    """
    single function to output the a and b matrices needed for the scatter matrices
    :param W_layer: gap
    :param Wg:
    :param V_layer: gap
    :param Vg:
    :return:
    """
    W_i = inv(W_layer)
    V_i = inv(V_layer)

    a = W_i @ Wg + V_i @ Vg
    b = W_i @ Wg - V_i @ Vg

    return a, b


def S_layer(A, B, d, k0, modes):
    """
    function to create scatter matrix in the ith layer of the uniform layer structure
    we assume that gap layers are used so we need only one A and one B
    :param A: function A =
    :param B: function B
    :param k0 #free -space wavevector magnitude (normalization constant) in Si Units
    :param Li #length of ith layer (in Si units)
    :param modes, eigenvalue matrix
    :return: S (4x4 scatter matrix) and Sdict, which contains the 2x2 block matrix as a dictionary
    """

    # sign convention (EMLAB is exp(-1i*k\dot r))
    X = np.diag(np.exp(-np.diag(modes)*d*k0))
    # CHECK: Check if this is expm

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
    """
    homogeneous layer is much simpler to do, so we will create an isolated module to deal with it
    :return:
    """
    assert type(Kx) == np.ndarray, 'not np.array'
    assert type(Ky) == np.ndarray, 'not np.array'

    N = len(Kx)
    I = np.identity(N)

    P = (e_r**-1)*np.block([[Kx*Ky, e_r*m_r*I-Kx**2], [Ky**2-m_r*e_r*I, -Ky*Kx]])
    Q = (e_r/m_r)*P

    diag = np.diag(Q)
    idx = np.nonzero(diag == 0)[0]
    if len(idx):
        # Adding pertub* to Q and pertub to Kz.
        Q[idx, idx] = np.conj(perturbation)
        print(wl, comment, 'non-invertible Q: adding perturbation')
        # print(Q.diagonal())

    W = np.eye(N*2)
    Kz2 = (m_r*e_r*I-Kx**2-Ky**2).astype('complex')  # arg is +kz^2
    # arg = -(m_r*e_r*I-Kx**2-Ky**2)  # arg is +kz^2
    # Kz = np.conj(np.sqrt(arg))  # conjugate enforces the negative sign convention (we also have to conjugate er and mur if they are complex)

    Kz = np.sqrt(Kz2)  # conjugate enforces the negative sign convention (we also have to conjugate er and mur if they are complex)
    Kz = np.conj(Kz)  # CHECK: conjugate?

    diag = np.diag(Kz)
    idx = np.nonzero(diag == 0)[0]
    if len(idx):
        Kz[idx, idx] = perturbation
        print(wl, comment, 'non-invertible Kz: adding perturbation')
        # print(Kz.diagonal())

    eigenvalues = block_diag(1j*Kz, 1j*Kz)  # determining the modes of ex, ey... so it appears eigenvalue order MATTERS...
    V = Q @ np.linalg.inv(eigenvalues)  # eigenvalue order is arbitrary (hard to compare with matlab


    # V = -1j*Q

    return W, V, Kz


def homogeneous_1D(Kx, n_index, m_r=1, pol=None, perturbation=1E-10, wl=None, comment=None):
    """
    efficient homogeneous 1D module
    :param Kx:
    :param e_r:
    :param m_r:
    :return:
    """

    e_r = n_index ** 2

    I = np.identity(len(Kx))

    W = I
    Q = (1 / m_r) * (e_r * m_r * I - Kx ** 2)
    # Q = Kx**2 - e_r * I

    diag = np.diag(Q)
    idx = np.nonzero(diag == 0)[0]
    if len(idx):
        # Adding pertub* to Q and pertub to Kz.
        Q[idx, idx] = np.conj(perturbation)
        print(wl, comment, 'non-invertible Q: adding perturbation')
        # print(Q.diagonal())

    Kz = np.sqrt(m_r*e_r*I-Kx**2)
    Kz = np.conj(Kz)  # CHECK: conjugate?

    # invertible check
    diag = np.diag(Kz)
    idx = np.nonzero(diag == 0)[0]
    if len(idx):
        Kz[idx, idx] = perturbation
        print(wl, comment, 'non-invertible Kz: adding perturbation')
        # print(Kz.diagonal())

    # CHECK: why this works...
    if pol:  # 0: TE, 1: TM
        Kz = Kz * (n_index ** 2)

    eigenvalues = -1j*Kz  # determining the modes of ex, ey... so it appears eigenvalue order MATTERS...
    V = Q @ np.linalg.inv(eigenvalues)  # eigenvalue order is arbitrary (hard to compare with matlab

    return W, V, Kz


def K_matrix_cubic_2D(beta_x, beta_y, k0, a_x, a_y, N_p, N_q):
    #    K_i = beta_i - pT1i - q T2i - r*T3i
    # but here we apply it only for cubic and tegragonal geometries in 2D
    """
    :param beta_x: input k_x,inc/k0
    :param beta_y: k_y,inc/k0; #already normalized...k0 is needed to normalize the 2*pi*lambda/a
            however such normalization can cause singular matrices in the homogeneous module (specifically with eigenvalues)
    :param T1:reciprocal lattice vector 1
    :param T2:
    :param T3:
    :return:
    """
    # (indexing follows (1,1), (1,2), ..., (1,N), (2,1),(2,2),(2,3)...(M,N) ROW MAJOR
    # but in the cubic case, k_x only depends on p and k_y only depends on q
    k_x = beta_x - 2*np.pi*np.arange(-N_p, N_p+1)/(k0*a_x)
    k_y = beta_y - 2*np.pi*np.arange(-N_q, N_q+1)/(k0*a_y)

    kx, ky = np.meshgrid(k_x, k_y)
    Kx = np.diag(kx.flatten())
    Ky = np.diag(ky.flatten())

    return Kx, Ky


def P_Q_kz(Kx, Ky, e_conv, mu_conv, oneover_E_conv, oneover_E_conv_i, E_i):
    '''
    r is for relative so do not put epsilon_0 or mu_0 here
    :param Kx: NM x NM matrix
    :param Ky:
    :param e_conv: (NM x NM) conv matrix
    :param mu_r:
    :return:
    '''
    argument = e_conv - Kx ** 2 - Ky ** 2
    Kz = np.conj(np.sqrt(argument.astype('complex')))
    # Kz = np.sqrt(argument.astype('complex'))  # CHECK: conjugate?

    # CHECK: confirm whether oneonver_E_conv is indeed not used
    # CHECK: Check sign of P and Q
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
    '''
        create a vector with a 1 corresponding to the 0th order
        #input P = 2*(num_ord_specified)+1
    '''
    fourier_grid = np.zeros((P,Q))
    fourier_grid[int(P/2), int(Q/2)] = 1
    # vector = np.zeros((P*Q,));
    #
    # #the index of the (0,0) element requires a conversion using sub2ind
    # index = int(P/2)*P + int(Q/2);
    vector = fourier_grid.flatten()
    return np.matrix(np.reshape(vector, (1,len(vector))))


def initial_conditions(K_inc_vector, theta, normal_vector, pte, ptm, P, Q):
    """
    :param K_inc_vector: whether it's normalized or not is not important...
    :param theta: angle of incience
    :param normal_vector: pointing into z direction
    :param pte: te polarization amplitude
    :param ptm: tm polarization amplitude
    :return:
    calculates the incident E field, cinc, and the polarization fro the initial condition vectors
    """
    # ate -> unit vector holding the out of plane direction of TE
    # atm -> unit vector holding the out of plane direction of TM
    # what are the out of plane components...(Ey and Hy)
    # normal_vector = [0,0,-1]; i.e. waves propagate down into the -z direction
    # cinc = Wr^-1@[Ex_inc, Ey_inc];

    if theta != 0:
        ate_vector = np.cross(K_inc_vector, normal_vector)
        ate_vector = ate_vector / (np.linalg.norm(ate_vector))
    else:
        ate_vector = np.array([0, 1, 0])

    atm_vector = np.cross(ate_vector, K_inc_vector)
    atm_vector = atm_vector / (np.linalg.norm(atm_vector))

    polarization = pte * ate_vector + ptm * atm_vector  # total E_field incident which is a 3 component vector (ex, ey, ez)
    E_inc = polarization
    # go from mode coefficients to FIELDS
    delta = delta_vector(2*P+1, 2*Q+1)

    # c_inc; #remember we ultimately solve for [Ex, Ey, Hx, Hy].
    e_src = np.hstack((polarization[0]*delta, polarization[1]*delta))
    e_src = np.matrix(e_src).T  # mode amplitudes of Ex, and Ey

    return E_inc, e_src, polarization


def RedhefferStar(SA, SB):  # SA and SB are both 2x2 block matrices;
    """
    RedhefferStar for arbitrarily sized 2x2 block matrices for RCWA
    :param SA: dictionary containing the four sub-blocks
    :param SB: dictionary containing the four sub-blocks,
    keys are 'S11', 'S12', 'S21', 'S22'
    :return:
    """

    assert type(SA) == dict, 'not dict'
    assert type(SB) == dict, 'not dict'

    # once we break every thing like this, we should still have matrices
    SA_11, SA_12, SA_21, SA_22 = SA['S11'], SA['S12'], SA['S21'], SA['S22']
    SB_11, SB_12, SB_21, SB_22 = SB['S11'], SB['S12'], SB['S21'], SB['S22']
    N = len(SA_11)  # SA_11 should be square so length is fine

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
    """
    this function assumes an RCWA implementation where all the scatter matrices are stored in a list
    and the global scatter matrix is constructed at the end
    :param scatter_list: list of scatter matrices of the form [Sr, S1, S2, ... , SN, ST]
    :return:
    """
    Sr = scatter_list[0]
    Sg = Sr
    for i in range(1, len(scatter_list)):
        Sg = RedhefferStar(Sg, scatter_list[i])
    return Sg

