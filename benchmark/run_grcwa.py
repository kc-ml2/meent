import grcwa
import numpy as np
import matplotlib.pyplot as plt

from meent.rcwa import RCWA


def run_grcwa():
    # Truncation order (actual number might be smaller)
    nG = 40
    # lattice constants
    L1 = [700, 0]
    L2 = [0, 700]
    # frequency and angles
    # freq = 1.
    # theta = np.pi / 6
    # phi = np.pi / 3
    theta = 1E-20
    phi = 1E-20
    # theta = 0
    # phi = 0
    # to avoid singular matrix, alternatively, one can add fictitious small loss to vacuum
    # Qabs = np.inf
    # freqcmp = freq*(1+1j/2/Qabs)
    # the patterned layer has a griding: Nx*Ny
    Nx = 1001
    Ny = 1001

    # now consider 3 layers: vacuum + patterned + vacuum
    ep0 = 1  # dielectric for layer 1 (uniform)
    epp = 3.48 ** 2  # dielectric for patterned layer
    epN = 2.  # dielectric for layer N (uniform)

    thick0 = 1  # thickness for vacuum layer 1
    thickp = 1120  # thickness of patterned layer
    thickN = 1

    # eps for patterned layer
    epgrid = np.ones((Nx, Ny), dtype=float)
    epgrid[:300, :] = epp

    wls = np.linspace(500, 2300, 100)

    Rall, Tall = [], []

    # Rall, Tall = np.array()

    for wl in wls:

        # setting up RCWA
        obj = grcwa.obj(nG, L1, L2, 1/wl, theta, phi, verbose=1)
        # input layer information
        obj.Add_LayerUniform(thick0, ep0)
        obj.Add_LayerGrid(thickp, Nx, Ny)
        obj.Add_LayerUniform(thickN, epN)
        obj.Init_Setup(Gmethod=1)

        # planewave excitation
        planewave = {'p_amp': 1, 's_amp': 0, 'p_phase': 0, 's_phase': 0}
        obj.MakeExcitationPlanewave(planewave['p_amp'], planewave['p_phase'], planewave['s_amp'], planewave['s_phase'],order = 0)
        # eps in patterned layer
        obj.GridLayer_geteps(epgrid.flatten())

        # compute reflection and transmission
        R_ord, T_ord = obj.RT_Solve(normalize=1, byorder=1)

        R, T = R_ord.sum(), T_ord.sum()

        # print('R=',R,', T=',T,', R+T=',R+T)
        Rall.append(R)
        Tall.append(T)

    plt.plot(wls, Rall)
    plt.plot(wls, Tall)
    plt.show()

    return Rall, Tall


if __name__ == '__main__':
    run_grcwa()
