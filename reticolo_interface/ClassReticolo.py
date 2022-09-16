import gym
import numpy as np
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


class ClassReticolo:

    def __init__(self, engine_type='octave'):
        super().__init__()

        if engine_type == 'octave':

            self.eng = octave
        elif engine_type == 'matlab':
            self.eng = matlab.engine.start_matlab()
        else:
            exit(1)

        # path that has file to run in octave
        m_path = os.path.dirname(__file__)
        self.eng.addpath(self.eng.genpath(m_path))

        os.makedirs('data', exist_ok=True)
        self.eff_file_path = 'data/' + '_eff_table.json'
        if Path(self.eff_file_path).exists():
            with open(self.eff_file_path, 'rb') as f:
                self.eff_table = json.load(f)
        else:
            self.eff_table = {}

        self.eff = 0

    def get_eff_of_structure(self, struct, wavelength, desired_angle):
        # img = struct
        # angle = desired_angle
        # effs = self.engine.Eval_Eff_1D(img, wavelength, angle)
        effs = self.eng.Eval_Eff_1D(struct, wavelength, desired_angle)
        print(effs)
        return effs

    def cal_reflectance(self, config, textures, profile, wavelength):

        R, T = self.eng.cal_reflectance(config, textures, profile, wavelength, nout=2)

        return R, T

    def generate_spectrum(self, incident_angle, detector_angle, wl_range, textures, profile):
        spectrum = np.zeros(len(wl_range))

        for i, wl in enumerate(wl_range):
            reflectivity = self.eng.cal_reflectance(textures, tuple(profile), wl, incident_angle, detector_angle)
            spectrum[i] = reflectivity

        return spectrum

    def generate_spectrum_loop(self, incident_angle, detector_angle, wl_begin, wl_end, wl_amount, textures, profile):

        spectrum = self.eng.cal_reflectance_loop(textures, tuple(profile), wl_begin, wl_end, wl_amount, incident_angle,
                                                 detector_angle)

        return spectrum
