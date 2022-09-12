import numpy as np
from scipy.linalg import block_diag
import cmath

#
# def homogeneous_module(Kx, Ky, e_r, m_r = 1):
#     """
#     homogeneous layer is much simpler to do, so we will create an isolated module to deal with it
#     :return:
#     """
#     assert type(Kx) == np.ndarray, 'not np.array'
#     assert type(Ky) == np.ndarray, 'not np.array'
#
#     N = len(Kx)
#     I = np.identity(N)
#     P = (e_r**-1)*np.block([[Kx*Ky, e_r*m_r*I-Kx**2], [Ky**2-m_r*e_r*I, -Ky*Kx]])
#     Q = (e_r/m_r)*P
#     W = np.identity(2*N)
#     arg = (m_r*e_r*I-Kx**2-Ky**2)  # arg is +kz^2
#     # arg = -(m_r*e_r*I-Kx**2-Ky**2)  # arg is +kz^2
#     arg = arg.astype('complex')
#     # Kz = np.conj(np.sqrt(arg))  # conjugate enforces the negative sign convention (we also have to conjugate er and mur if they are complex)
#     Kz = np.sqrt(arg)  # conjugate enforces the negative sign convention (we also have to conjugate er and mur if they are complex)
#     eigenvalues = block_diag(-1j*Kz, -1j*Kz)  # determining the modes of ex, ey... so it appears eigenvalue order MATTERS...
#     # W is just identity matrix
#
#     # # Singular. PERTURBATION.
#     # a = np.diag(eigenvalues)
#     # b = np.nonzero(a == 0)  # TODO:possibility of overflow?
#     # eigenvalues[b, b] = np.inf
#
#     V = Q@np.linalg.inv(eigenvalues)  # eigenvalue order is arbitrary (hard to compare with matlab
#     # alternative V with no inverse
#     # V = np.matmul(np.linalg.inv(P),np.matmul(Q,W)); apparently, this fails because P is singular
#     return W, V, Kz
def homogeneous_module(Kx, Ky, e_r, m_r = 1):
    """
    homogeneous layer is much simpler to do, so we will create an isolated module to deal with it
    :return:
    """
    assert type(Kx) == np.ndarray, 'not np.array'
    assert type(Ky) == np.ndarray, 'not np.array'
    j = cmath.sqrt(-1)
    N = len(Kx)
    I = np.identity(N)
    P = (e_r**-1)*np.block([[Kx*Ky, e_r*m_r*I-Kx**2], [Ky**2-m_r*e_r*I, -Ky*Kx]])
    Q = (e_r/m_r)*P
    W = np.identity(2*N)
    arg = (m_r*e_r*I-Kx**2-Ky**2)  # arg is +kz^2
    arg = arg.astype('complex')
    Kz = np.conj(np.sqrt(arg))  # conjugate enforces the negative sign convention (we also have to conjugate er and mur if they are complex)
    eigenvalues = block_diag(j*Kz, j*Kz)  # determining the modes of ex, ey... so it appears eigenvalue order MATTERS...
    # W is just identity matrix



    V = Q@np.linalg.inv(eigenvalues)  # eigenvalue order is arbitrary (hard to compare with matlab
    # alternative V with no inverse
    # V = np.matmul(np.linalg.inv(P),np.matmul(Q,W)); apparently, this fails because P is singular
    return W, V, Kz


    with open('test1.npy', 'wb') as f:
        np.save(f, Kx)
        np.save(f, Ky)

def homogeneous_1D(Kx, k0, n_index, m_r=1, tt=None):
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
    Q = Kx**2 - e_r * I

    Kz2 = (m_r*e_r*I-Kx**2)

    Kz = np.sqrt(Kz2)

    if tt:  # 0: TE, 1: TM
        Kz = Kz * (n_index ** 2)

    eigenvalues = -1j*Kz  # determining the modes of ex, ey... so it appears eigenvalue order MATTERS...

    # Singular. PERTURBATION.
    a = np.diag(eigenvalues)
    b = np.nonzero(a == 0)  # TODO:possibility of overflow?
    eigenvalues[b, b] = np.inf

    V = Q @ np.linalg.inv(eigenvalues)  # eigenvalue order is arbitrary (hard to compare with matlab

    return W, V, Kz



