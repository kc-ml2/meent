import grcwa
import numpy as np
import matplotlib.pyplot as plt

from meent.on_numpy.emsolver._base import _BaseRCWA


class GRCWA:
    def __init__(self, grating_type=0, n_I=1., n_II=1., theta=0., phi=0., psi=0, fourier_order=40, period=(100,),
                 wavelength=900., pol=1, ucell=None, thickness=(325,), *args, **kwargs):

        # super().__init__(grating_type)

        # Truncation order (actual number might be smaller)
        self.fourier_order = fourier_order

        # lattice constants
        self.pol = pol
        self.period = np.array(period)
        if not self.period.shape:
            self.L1 = [self.period, 0]
            self.L2 = [0, self.period]
        elif len(self.period) == 1:
            self.L1 = [self.period[0], 0]
            self.L2 = [0, self.period[0]]
        else:
            self.L1 = [self.period[0], 0]
            self.L2 = [0, self.period[1]]

        self.theta = theta * np.pi / 180
        self.phi = phi * np.pi / 180

        self.Nx = 1001
        self.Ny = 1001

        # now consider 3 layers: vacuum + patterned + vacuum
        self.n_I = n_I  # dielectric for layer 1 (uniform)
        self.epp = 3.48 ** 2  # dielectric for patterned layer
        self.n_II = n_II  # dielectric for layer N (uniform)

        self.thick0 = 1  # thickness for vacuum layer 1
        self.thickp = np.array(thickness)  # thickness of patterned layer
        self.thickN = 1

        self.ucell = ucell

        self.wavelength = wavelength

    def run(self, pattern=None):

        # setting up RCWA
        obj = grcwa.obj(self.fourier_order, self.L1, self.L2, 1/self.wavelength, self.theta, self.phi, verbose=0)
        # input layer information
        obj.Add_LayerUniform(self.thick0, self.n_I)
        obj.Add_LayerGrid(self.thickp, self.Nx, self.Ny)
        obj.Add_LayerUniform(self.thickN, self.n_II)
        obj.Init_Setup(Gmethod=1)

        # planewave excitation
        planewave = {'p_amp': self.pol, 's_amp': 1-self.pol, 'p_phase': 0, 's_phase': 0}
        obj.MakeExcitationPlanewave(planewave['p_amp'], planewave['p_phase'],
                                    planewave['s_amp'], planewave['s_phase'], order=0)

        # eps in patterned layer
        obj.GridLayer_geteps(self.ucell.flatten())

        # compute reflection and transmission
        de_ri, de_ti = obj.RT_Solve(normalize=1, byorder=1)

        # de_ti, de_ti = de_ri.sum(), de_ti.sum()

        # print('R=', R, ', T=', T, ', R+T=', R+T)

        return de_ri, de_ti


if __name__ == '__main__':
    pol = 1
    fourier_order = 40

    # lattice constants
    period = 700

    theta = 1E-20
    phi = 1E-20
    # the patterned layer has a griding: Nx*Ny
    Nx = 1001
    Ny = 1001

    # now consider 3 layers: vacuum + patterned + vacuum
    n_I = 1  # dielectric for layer 1 (uniform)
    n_II = 1  # dielectric for layer N (uniform)

    thickness = 1120  # thickness of patterned layer

    epp = 3.48 ** 2  # dielectric for patterned layer
    # eps for patterned layer
    ucell = np.ones((Nx, Ny), dtype=float)
    ucell[:300, :] = epp

    wavelength = 900.

    res = GRCWA(grating_type=0, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=0, fourier_order=fourier_order,
                period=period, wavelength=wavelength, pol=pol, ucell=ucell, thickness=thickness)
    de_ri, de_ti = res.run()
    print(de_ri.sum() + de_ti.sum())
    print(de_ri)
    print(de_ti)
