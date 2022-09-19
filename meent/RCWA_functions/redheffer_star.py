import numpy as np
import cmath
from numpy.linalg import solve as bslash
from numpy.linalg import inv, pinv
# TODO: try pinv


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
