import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import matlab.engine
except:
    pass

os.environ['OCTAVE_EXECUTABLE'] = '/opt/homebrew/bin/octave-cli'
from oct2py import octave

from meent._base import Base


class Reticolo(Base):

    def __init__(self, grating_type=0,
                 n_I=1., n_II=1., theta=0., phi=0., fourier_order=40, period=(100,),
                 wls=np.linspace(900, 900, 1), pol=1,
                 textures=None, profile=None,
                 engine_type='octave'):
        super().__init__(grating_type)

        if engine_type == 'octave':
            self.eng = octave
        elif engine_type == 'matlab':
            self.eng = matlab.engine.start_matlab()
        else:
            raise ValueError

        # path that has file to run in octave
        m_path = os.path.dirname(__file__)
        self.eng.addpath(self.eng.genpath(m_path))

        self.grating_type = grating_type
        self.n_I = n_I
        self.n_II = n_II

        self.theta = theta * np.pi / 180
        self.phi = phi * np.pi / 180
        self.pol = pol  # TE 0, TM 1

        self.fourier_order = fourier_order
        self.ff = 2 * self.fourier_order + 1

        self.period = period

        self.wls = wls
        self.textures = textures
        self.profile = profile

        self.init_spectrum_array()

    def run(self):

        for i, wl in enumerate(self.wls):
            de_ri, de_ti = self.eng.run_reticolo(self.pol, self.theta, self.period, self.n_I, self.fourier_order,
                                                 self.textures, self.profile, wl, nout=2)

            self.save_spectrum_array(de_ri, de_ti, i)

        return self.spectrum_r, self.spectrum_t

    def run_acs(self, pattern, wl, deflected_angle):
        a, b = self.eng.Eval_Eff_1D(pattern, wl, deflected_angle, self.fourier_order, nout=2)
        return a, b

    def run_acs_loop_wavelength(self, pattern, deflected_angle, wls=None):
        if wls is None:
            wls = self.wls
        else:
            self.wls = wls  # TODO: handle better.

        self.init_spectrum_array()

        for i, wl in enumerate(wls):
            _, de_ti, de_ri = self.eng.Eval_Eff_1D(pattern, wl, deflected_angle, self.fourier_order, nout=3)
            self.save_spectrum_array(de_ri, de_ti, i)

        return self.spectrum_r, self.spectrum_t


if __name__ == '__main__':
    Nx = 1001
    Ny = 1001

    n_I = 1
    n_si = 3.48
    n_II = 1
    theta = 1E-10
    phi = 0
    fourier_order = 40

    period = 700
    wls = np.linspace(500, 2300, 100)
    pol = 1

    thickness = 1120
    # eps for patterned layer
    pattern = np.ones(Nx, dtype=float)
    grid = np.linspace(0, period, 1001)
    pattern[:300] = n_si

    textures = [1, [grid, pattern], 1]

    profile = np.array([[0, thickness, 0], [1, 2, 3]])

    AA = Reticolo(grating_type=0,
                  n_I=n_I, n_II=n_II, theta=theta, phi=phi, fourier_order=fourier_order, period=period,
                  wls=wls, pol=pol,
                  textures=textures, profile=profile,
                  engine_type='octave')

    refl, tran = AA.run()
    AA.plot()
    pass
