import grcwa
import numpy as np
import matplotlib.pyplot as plt


class GRCWA:
    def __init__(self, n_I=1., n_II=1., theta=0., phi=0., fourier_order=40, period=(100,),
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

        self.theta = theta
        self.phi = phi

        # now consider 3 layers: vacuum + patterned + vacuum
        self.n_I = n_I  # dielectric for layer 1 (uniform)
        self.epp = 3.48 ** 2  # dielectric for patterned layer
        self.n_II = n_II  # dielectric for layer N (uniform)

        self.thick0 = 1  # thickness for vacuum layer 1
        self.thickp = np.array(thickness)  # thickness of patterned layer
        self.thickN = 1

        self.ucell = ucell**2
        self.Nx = ucell.shape[2]
        self.Ny = ucell.shape[1]

        self.wavelength = wavelength

    def run(self, pattern=None):

        # setting up RCWA
        obj = grcwa.obj(self.fourier_order, self.L1, self.L2, 1/self.wavelength, self.theta, self.phi, verbose=0)
        # input layer information
        obj.Add_LayerUniform(self.thick0, self.n_I)
        for thickness in self.thickp:
            obj.Add_LayerGrid(thickness, self.Ny, self.Nx)
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

    option = {
        'grating_type': 0,
        'pol': 1,
        'n_top': 1,
        'n_bot': 1,
        'theta': 1,
        'phi': 1,
        'wavelength': 1,
        'fourier_order': 1,
        'thickness': [1000, 300],
        'period': [1000],
        'fourier_type': 1,
        'ucell': np.array(
            [
                [[3.1, 1.1, 1.2, 1.6, 3.1]],
                [[3, 3, 1, 1, 1]],
            ]
        ),
    }

    res = GRCWA(**option)

    de_ri, de_ti = res.run()
    print(de_ri.sum() + de_ti.sum())
    print(de_ri)
    print(de_ti)
