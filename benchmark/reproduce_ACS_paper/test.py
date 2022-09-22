import time
import numpy as np

from benchmark.interface.Reticolo import Reticolo
from examples.rcwa.JLAB import JLABCode

wavelength = 900
deflected_angle = 60
pattern = np.array([ 1.,  1.,  1., -1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1., -1.,  1.,  1.,
  1., -1.,  1.,  1.,  1., -1.,  1.,  1.,  1., -1.,  1.,  1.,  1.,  1.,  1.,  1.,  1., -1.,
 -1.,  1.,  1., -1., -1.,  1.,  1.,  1.,  1., -1.,  1., -1.,  1.,  1.,  1., -1.,  1.,  1.,
 -1.,  1.,  1.,  1.,  1., -1.,  1.,  1.,  1.,  1.])


wls = np.linspace(898, 902, 100)
fourier_order = 40

t0 = time.time()

mnt = JLABCode(wls=wls, fourier_order=fourier_order)
mnt.reproduce_acs_loop_wavelength(pattern, deflected_angle)
# mnt.plot(marker='x')
t1 = time.time()

t2 = time.time()

reti = Reticolo(wls=wls, fourier_order=fourier_order)
reti.run_acs_loop_wavelength(pattern, deflected_angle)
# reti.plot(marker='x')
t3 = time.time()

print(t1 - t0)
print(t3 - t2)


wls = np.linspace(898, 902, 100)
fourier_order = 80

t0 = time.time()

mnt = JLABCode(wls=wls, fourier_order=fourier_order)
mnt.reproduce_acs_loop_wavelength(pattern, deflected_angle)
# mnt.plot(marker='x')
t1 = time.time()

t2 = time.time()

reti = Reticolo(wls=wls, fourier_order=fourier_order)
reti.run_acs_loop_wavelength(pattern, deflected_angle)
# reti.plot(marker='x')
t3 = time.time()

print(t1 - t0)
print(t3 - t2)


wls = np.linspace(898, 902, 100)
fourier_order = 160

t0 = time.time()

mnt = JLABCode(wls=wls, fourier_order=fourier_order)
mnt.reproduce_acs_loop_wavelength(pattern, deflected_angle)
# mnt.plot(marker='x')
t1 = time.time()

t2 = time.time()

reti = Reticolo(wls=wls, fourier_order=fourier_order)
reti.run_acs_loop_wavelength(pattern, deflected_angle)
# reti.plot(marker='x')
t3 = time.time()

print(t1 - t0)
print(t3 - t2)
