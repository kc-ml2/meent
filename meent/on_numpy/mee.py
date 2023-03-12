import numpy as np

from .modeler.modeling import ModelingNumpy
from .emsolver.rcwa import RCWANumpy


class MeeNumpy(ModelingNumpy, RCWANumpy):

    def __init__(self, *args, **kwargs):
        ModelingNumpy.__init__(self, *args, **kwargs)
        RCWANumpy.__init__(self, *args, **kwargs)

    def spectrum(self, wavelength_list):
        if self.grating_type in (0, 1):
            de_ri_list = np.zeros((len(wavelength_list), self.fourier_order))
            de_ti_list = np.zeros((len(wavelength_list), self.fourier_order))
        else:
            de_ri_list = np.zeros((len(wavelength_list), self.ff, self.ff))
            de_ti_list = np.zeros((len(wavelength_list), self.ff, self.ff))

        for i, wavelength in enumerate(wavelength_list):
            de_ri, de_ti = self.conv_solve(wavelength=wavelength)
            de_ri_list[i] = de_ri
            de_ti_list[i] = de_ti

        return de_ri_list, de_ti_list
