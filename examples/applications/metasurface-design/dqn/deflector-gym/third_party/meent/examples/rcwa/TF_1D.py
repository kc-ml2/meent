import matplotlib.pyplot as plt
import numpy as np

pi = np.pi

n_I = 1
n_II = 1

theta = 0

wls = np.linspace(0.5, 2.3, 400)

spectrum_r, spectrum_t = [], []
n_l = 3.48
d_l = 0.46

for wl in wls:
    k0 = 2 * np.pi / wl
    kx = k0 * n_I * np.sin(theta)
    k_I_z = k0 * n_I * np.cos(theta)
    k_II_z = k0 * (n_II**2 - n_I**2 * np.sin(theta)**2)**0.5

    gamma = 1j*(n_l**2 - n_I**2 * np.sin(theta)**2)**0.5

    A = np.array([[1,1], [gamma, -gamma]])

    a, b = np.linalg.inv(A) @ np.array([[1], [1j * k_II_z / k0]]).flatten()

    f, g = np.array([[a + b*np.exp(-2*k0*gamma*d_l)],
                     [gamma*(a - b*np.exp(-2*k0*gamma*d_l))]]).flatten()

    T1 = (g + 1j*(k_I_z/k0)*f)**-1 * 2j*(k_I_z/k0)
    R = f * T1 - 1

    T = np.exp(-k0*gamma*d_l) * T1

    DEr = R * np.conj(R) * np.real(k_I_z / (k0 * n_I * np.cos(theta)))
    DEt = T * np.conj(T) * np.real(k_II_z / (k0 * n_I * np.cos(theta)))

    spectrum_r.append(DEr.real)
    spectrum_t.append(DEt.real)

plt.plot(wls, spectrum_r)
plt.plot(wls, spectrum_t)
plt.show()
pass
