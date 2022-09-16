import numpy as np
from numpy.linalg import inv, pinv
# TODO: try pseudo-inverse?


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
    X = np.diag(np.exp(-np.diag(modes)*d*k0))  # never use expm

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
