import numpy as np
from numpy.linalg import solve, inv
from numpy.linalg import solve as bslash

## description
# kx, ky are normalized wavevector MATRICES NOW (by k0)
# matrices because we have expanded in some number of spatial harmonics
# e_r and m_r do not have e0 or mu0 in them
# presently we assume m_r is homogeneous 1
##====================================##

def Q_matrix(Kx, Ky, E_conv, mu_conv, oneover_E_conv, oneover_E_conv_i):
    '''
    pressently assuming non-magnetic material so mu_conv = I
    :param Kx: now a matrix (NM x NM)
    :param Ky: now a matrix
    :param e_conv: (NM x NM) matrix containing the 2d convmat
    :return:
    '''

    assert type(Kx) == np.ndarray, 'not array'
    assert type(Ky) == np.ndarray, 'not array'
    assert type(E_conv) == np.ndarray, 'not array'

    # Kx , Ky = np.conj(Kx), np.conj(Ky)

    # return np.block([[Kx @ bslash(mu_conv,Ky),  E_conv - Kx @ bslash(mu_conv, Kx)],
    #                                      [Ky @ bslash(mu_conv, Ky)  - E_conv, -Ky @ bslash(mu_conv, Kx)]]);
    Q = np.block([
        [Kx @ inv(mu_conv) @ Ky, -Kx @ inv(mu_conv) @ Kx + E_conv],
        [-oneover_E_conv_i + Ky @ inv(mu_conv) @ Ky, -Ky @ inv(mu_conv) @ Kx]
    ])
    # Q = np.block([
    #     [-Kx @ inv(mu_conv) @ Ky, Kx @ inv(mu_conv) @ Kx - E_conv],
    #     [oneover_E_conv_i - Ky @ inv(mu_conv) @ Ky, Ky @ inv(mu_conv) @ Kx]
    # ])

    # return np.block([[-Kx @ bslash(mu_conv,Ky),  -E_conv + Kx @ bslash(mu_conv, Kx)],
    #                                      [-Ky @ bslash(mu_conv, Ky) + E_conv, Ky @ bslash(mu_conv, Kx)]]);

    return Q

def P_matrix(Kx, Ky, E_conv, mu_conv, oneover_E_conv, oneover_E_conv_i, E_i):
    assert type(Kx) == np.ndarray, 'not array'
    assert type(Ky) == np.ndarray, 'not array'
    assert type(E_conv) == np.ndarray, 'not array'
    # Kx , Ky = np.conj(Kx), np.conj(Ky)

    P = np.block([
        [Kx @ E_i @ Ky, -Kx @ E_i @ Kx + mu_conv],
        [Ky @ E_i @ Ky - mu_conv,  -Ky @ E_i @ Kx]
    ])

    # P = np.block([
    #     [-Kx @ inv(E_conv) @ Ky, Kx @ inv(E_conv) @ Kx - mu_conv],
    #     [mu_conv - Ky @ inv(E_conv) @ Ky,  Ky @ inv(E_conv) @ Kx]
    # ])

    # P = np.block([[Kx @ bslash(E_conv, Ky),  mu_conv - Kx @ bslash(E_conv,Kx)],
    #               [Ky @ bslash(E_conv, Ky) - mu_conv,  -Ky @ bslash(E_conv,Kx)]]);
    # P = np.block([[-Kx @ bslash(E_conv, Ky),  -mu_conv + Kx @ bslash(E_conv,Kx)],
    #               [-Ky @ bslash(E_conv, Ky) + mu_conv,  Ky @ bslash(E_conv,Kx)]]);

    return P



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
    # Kz = np.sqrt(argument.astype('complex'))
    q = Q_matrix(Kx, Ky, e_conv, mu_conv, oneover_E_conv, oneover_E_conv_i)
    p = P_matrix(Kx, Ky, e_conv, mu_conv, oneover_E_conv, oneover_E_conv_i, E_i)

    return p, q, Kz;

## simple test case;

