import time
import numpy as np

from ml2rcwa._base import _BaseRCWA


class RCWA(_BaseRCWA):
    def __init__(self, grating_type=0, n_I=1., n_II=1., theta=0, phi=0, psi=0, fourier_order=40, period=(100,),
                 wls=np.linspace(900, 900, 1), pol=1, patterns=None, thickness=(325,), algo='TMM'):

        super().__init__(grating_type, n_I, n_II, theta, phi, psi, fourier_order, period, wls, pol, patterns,
                         thickness, algo)
        self.spectrum_r, self.spectrum_t = None, None
        self._init_spectrum()

    def _init_spectrum(self):
        if self.grating_type in (0, 1):
            self.spectrum_r = np.ndarray((len(self.wls), 2 * self.fourier_order + 1))
            self.spectrum_t = np.ndarray((len(self.wls), 2 * self.fourier_order + 1))
        elif self.grating_type == 2:
            self.spectrum_r = np.ndarray((len(self.wls), self.ff, self.ff))
            self.spectrum_t = np.ndarray((len(self.wls), self.ff, self.ff))
        else:
            raise ValueError

    def run_single(self, wl):
        E_conv_all, oneover_E_conv_all = self.get_permittivity_map(wl)

        if self.grating_type == 0:
            de_ri, de_ti = self.lalanne_1d(wl, E_conv_all, oneover_E_conv_all)
        elif self.grating_type == 1:
            # de_ri, de_ti = self.lalanne_1d_conical()  # TODO: implement
            de_ri = de_ti = None
        elif self.grating_type == 2:
            de_ri, de_ti = self.lalanne_2d(wl, E_conv_all, oneover_E_conv_all)
        else:
            raise ValueError

        return de_ri, de_ti

    def loop_wavelength(self, wavelength_array=None):

        if wavelength_array is not None:
            self.wls = wavelength_array
            self._init_spectrum()

        for i, wl in enumerate(self.wls):

            de_ri, de_ti = self.run_single(wl)

            self.spectrum_r[i] = de_ri
            self.spectrum_t[i] = de_ti

        return self.spectrum_r, self.spectrum_t


if __name__ == '__main__':
    grating_type = 2
    pol = 0

    n_r = 1.45  # glass
    n_t = 1

    theta = 0
    phi = 0
    psi = 0 if pol else 90

    wls = np.linspace(500, 2300, 10)

    if grating_type == 2:
        period = [700, 700]
        fourier_order = 2

    else:
        period = [700]
        fourier_order = 10

    # permittivity in grating layer
    patterns = [[3.48, 1, 0.3], [3.48, 1, 0.3]]  # n_ridge, n_groove, fill_factor
    thickness = [460, 660]

    AA = RCWA(grating_type=grating_type, pol=pol, n_I=n_r, n_II=n_t, theta=theta, phi=phi, psi=psi,
              fourier_order=fourier_order, wls=wls, period=period, patterns=patterns, thickness=thickness)
    t0 = time.perf_counter()

    a, b = AA.loop_wavelength()
    AA.plot()

    print(time.perf_counter() - t0)

