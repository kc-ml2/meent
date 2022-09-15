import numpy as np
from scipy.linalg import block_diag


def homogeneous_module(Kx, Ky, e_r, m_r=1, perturbation=1E-16, wl=None, comment=None):
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

    # Q = (m_r**-1)*np.block([[Kx*Ky, e_r*m_r*I-Kx**2], [Ky**2-m_r*e_r*I, -Kx*Ky]])

    diag = np.diag(Q)
    idx = np.nonzero(diag == 0)[0]
    if len(idx):
        # Adding pertub* to Q and pertub to Kz.
        # TODO: check why this works.
        # TODO: make imaginary part sign consistent
        Q[idx, idx] = np.conj(perturbation)
        print(wl, comment, 'non-invertible Q: adding perturbation')
        # print(Q.diagonal())

    W = np.eye(N*2)
    Kz2 = (m_r*e_r*I-Kx**2-Ky**2).astype('complex')  # arg is +kz^2
    # arg = -(m_r*e_r*I-Kx**2-Ky**2)  # arg is +kz^2
    # Kz = np.conj(np.sqrt(arg))  # conjugate enforces the negative sign convention (we also have to conjugate er and mur if they are complex)

    Kz = np.sqrt(Kz2)  # conjugate enforces the negative sign convention (we also have to conjugate er and mur if they are complex)
    Kz = np.conj(Kz)  # TODO: conjugate?

    diag = np.diag(Kz)
    idx = np.nonzero(diag == 0)[0]
    if len(idx):
        Kz[idx, idx] = perturbation
        print(wl, comment, 'non-invertible Kz: adding perturbation')
        # print(Kz.diagonal())

    eigenvalues = block_diag(1j*Kz, 1j*Kz)  # determining the modes of ex, ey... so it appears eigenvalue order MATTERS...

    V = Q @ np.linalg.inv(eigenvalues)  # eigenvalue order is arbitrary (hard to compare with matlab
    # alternative V with no inverse
    # V = np.matmul(np.linalg.inv(P),np.matmul(Q,W)); apparently, this fails because P is singular

    # Q = np.block([[Kx*Ky, I+Ky**2], [-Kx**2-I, -Kx*Ky]])
    # Q = np.block([[Kx*Ky, I-Kx**2], [Ky**2-I, -Kx*Ky]])
    # Q = (m_r**-1)*np.block([[Kx*Ky, e_r*m_r*I-Kx**2], [Ky**2-m_r*e_r*I, -Ky*Kx]])



    # V = -1j*Q

    return W, V, Kz
# def homogeneous_module(Kx, Ky, e_r, m_r = 1):
#     """
#     homogeneous layer is much simpler to do, so we will create an isolated module to deal with it
#     :return:
#     """
#     assert type(Kx) == np.ndarray, 'not np.array'
#     assert type(Ky) == np.ndarray, 'not np.array'
#     j = cmath.sqrt(-1)
#     N = len(Kx)
#     I = np.identity(N)
#     P = (e_r**-1)*np.block([[Kx*Ky, e_r*m_r*I-Kx**2], [Ky**2-m_r*e_r*I, -Ky*Kx]])
#     Q = (e_r/m_r)*P
#     W = np.identity(2*N)
#     arg = (m_r*e_r*I-Kx**2-Ky**2)  # arg is +kz^2
#     arg = arg.astype('complex')
#     Kz = np.conj(np.sqrt(arg))  # conjugate enforces the negative sign convention (we also have to conjugate er and mur if they are complex)
#     eigenvalues = block_diag(j*Kz, j*Kz)  # determining the modes of ex, ey... so it appears eigenvalue order MATTERS...
#     # W is just identity matrix
#
#
#
#     V = Q@np.linalg.inv(eigenvalues)  # eigenvalue order is arbitrary (hard to compare with matlab
#     # alternative V with no inverse
#     # V = np.matmul(np.linalg.inv(P),np.matmul(Q,W)); apparently, this fails because P is singular
#     return W, V, Kz
#
#
#     with open('test1.npy', 'wb') as f:
#         np.save(f, Kx)
#         np.save(f, Ky)

def homogeneous_1D(Kx, n_index, m_r=1, pol=None, perturbation=1E-20*(1-1j), wl=None, comment=None):
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
    Q = e_r * m_r * I - Kx ** 2
    # Q = Kx**2 - e_r * I

    diag = np.diag(Q)
    idx = np.nonzero(diag == 0)[0]
    if len(idx):
        # Adding pertub* to Q and pertub to Kz.
        # TODO: check why this works.
        # TODO: make imaginary part sign consistent
        Q[idx, idx] = np.conj(perturbation)
        print(wl, comment, 'non-invertible Q: adding perturbation')
        # print(Q.diagonal())

    Kz = np.sqrt(m_r*e_r*I-Kx**2)
    Kz = np.conj(Kz)  # TODO: conjugate?

    # TODO: check Singular or ill-conditioned; spread this to whole code
    # invertible check
    diag = np.diag(Kz)
    idx = np.nonzero(diag == 0)[0]
    if len(idx):
        Kz[idx, idx] = perturbation
        print(wl, comment, 'non-invertible Kz: adding perturbation')
        # print(Kz.diagonal())

    # TODO: why this works...
    if pol:  # 0: TE, 1: TM
        Kz = Kz * (n_index ** 2)

    eigenvalues = -1j*Kz  # determining the modes of ex, ey... so it appears eigenvalue order MATTERS...

    # # TODO:Singular. PERTURBATION.
    # try:
    #     V = Q @ np.linalg.inv(eigenvalues)  # eigenvalue order is arbitrary (hard to compare with matlab
    # except:
    #     print('Singular matrix in eigenvalues in homogeneous_module')
    #     a = np.diag(eigenvalues)
    #     b = np.nonzero(a == 0)  # TODO:possibility of overflow?
    #     eigenvalues_1 = eigenvalues.copy()
    #
    #     eigenvalues_1[b, b] = 1E-10
    #     V = Q @ np.linalg.inv(eigenvalues_1)  # eigenvalue order is arbitrary (hard to compare with matlab
    #     print(1)

    V = Q @ np.linalg.inv(eigenvalues)  # eigenvalue order is arbitrary (hard to compare with matlab

    return W, V, Kz



