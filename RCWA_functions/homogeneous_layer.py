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


    # V = -1j*Q

    return W, V, Kz


def homogeneous_1D(Kx, n_index, m_r=1, pol=None, perturbation=1E-20*(1+1j), wl=None, comment=None):
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
    V = Q @ np.linalg.inv(eigenvalues)  # eigenvalue order is arbitrary (hard to compare with matlab

    return W, V, Kz



