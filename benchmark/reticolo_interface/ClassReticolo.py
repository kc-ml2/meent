import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import matlab.engine
except:
    pass

from pathlib import Path
import _pickle as json
import os

os.environ['OCTAVE_EXECUTABLE'] = '/opt/homebrew/bin/octave-cli'
from oct2py import octave


class ClassReticolo():

    def __init__(self, n_cells, wavelength, desired_angle, engine_type='octave'):

        if engine_type == 'octave':
            self.eng = octave
        elif engine_type == 'matlab':
            self.eng = matlab.engine.start_matlab()
        else:
            raise ValueError

        # path that has file to run in octave
        m_path = os.path.dirname(__file__)
        self.eng.addpath(self.eng.genpath(m_path))

        os.makedirs('data', exist_ok=True)

        self.eff = 0
        self.n_cells = n_cells
        self.wavelength = self.eng.double([wavelength])

        self.desired_angle = self.eng.double([desired_angle])
        self.struct = np.ones(self.n_cells)

    def cal_reflectance(self, textures, profile, wavelength, theta, deflect_angle):
        R, T = self.eng.cal_reflectance(0, 1, 1, 300, 4, textures, profile,1, wavelength, nout=2)

        return R, T

    def generate_spectrum(self, incident_angle, detector_angle, wl_range, textures, profile):
        spectrum = np.zeros(len(wl_range))

        for i, wl in enumerate(wl_range):
            R = self.eng.cal_reflectance(0, 60, 1, 1.25, 300, 4, textures, profile, wl)
            spectrum[i] = R

        print(spectrum)
        return spectrum

    # def generate_spectrum_loop(self, incident_angle, detector_angle, wl_begin, wl_end, wl_amount, textures, profile):
    #
    #     spectrum = self.eng.cal_reflectance_loop(textures, tuple(profile), wl_begin, wl_end, wl_amount, incident_angle,
    #                                              detector_angle)
    #
    #     return spectrum


if __name__ == '__main__':
    AA = ClassReticolo(10, 900, 60)
    theta = 0
    deflect_angle = 60
    wls = np.linspace(500, 800, 10)
    Nx = 1001
    Ny = 1001

    # now consider 3 layers: vacuum + patterned + vacuum
    ep0 = 1  # dielectric for layer 1 (uniform)
    epp = 3.48 ** 2 # dielectric for patterned layer
    epN = 2.  # dielectric for layer N (uniform)

    thick0 = 1  # thickness for vacuum layer 1
    thickp = 1120  # thickness of patterned layer
    thickN = 1

    # eps for patterned layer
    epgrid = np.ones(Nx, dtype=float)
    grid = np.linspace(-100, 100, 1001)
    epgrid[:300] = epp
    textures = [1, [grid, epgrid], 1.45]

    profile = np.array([[0, 300, 0], [1, 2, 3]])

    AA.generate_spectrum(theta, deflect_angle, wls, textures, profile)
