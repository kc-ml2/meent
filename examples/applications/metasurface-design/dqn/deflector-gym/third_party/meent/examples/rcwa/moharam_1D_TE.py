import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import toeplitz


def E_conv_1D_analytic(fourier_order, patterns, period):
    eps = np.zeros(2 * fourier_order + 1).astype('complex')
    E_conv_all = []

    for i, (n_rd, n_gr, fill_factor) in enumerate(patterns):

        for i, order in enumerate(range(-fourier_order, fourier_order + 1)):
            if order == 0: continue
            eps[i] = (n_rd ** 2 - n_gr ** 2) * np.sin(np.pi * order * fill_factor) / (np.pi * order)
            eps[i] *= np.exp(1j * (2 * np.pi * order) / period)

        eps[fourier_order] = (n_rd ** 2 * fill_factor + n_gr ** 2 * (1 - fill_factor))

        E = toeplitz(np.concatenate((eps[fourier_order:], eps[:fourier_order])))
        E_conv_all.append(E)

    return E_conv_all


pi = np.pi

n_I = 1
n_II = 1

theta = 0 * pi / 180

fourier_order = 10
period = 0.7

wls = np.linspace(0.5, 2.3, 400)

spectrum_r, spectrum_t = [], []

# permittivity in grating layer
patterns = [[3.48, 1, 0.3], [3.48, 1, 0.3]]  # n_ridge, n_groove, fill_factor

thickness = [0.46, 0.66]

E_conv_all = E_conv_1D_analytic(fourier_order, patterns, period)

fourier_indices = np.arange(-fourier_order, fourier_order + 1)

delta_i0 = np.zeros(2 * fourier_order + 1)
delta_i0[fourier_order] = 1

for wl in wls:
    k0 = 2 * np.pi / wl

    kx_vector = k0*(n_I*np.sin(theta) - fourier_indices * (wl/period)).astype('complex')

    k_I_z = (k0**2 * n_I ** 2 - kx_vector**2)**0.5
    k_II_z = (k0**2 * n_II ** 2 - kx_vector**2)**0.5

    k_I_z = k_I_z.conjugate()
    k_II_z = k_II_z.conjugate()

    Y_I = np.diag(k_I_z / k0)
    Y_II = np.diag(k_II_z / k0)

    Kx = np.diag(kx_vector/k0)

    f = np.eye(2*fourier_order+1)
    g = 1j*Y_II

    T = np.eye(2*fourier_order+1)

    for E_conv, d in zip(E_conv_all[::-1], thickness[::-1]):

        A = Kx**2 - E_conv
        eigenvalues, W = np.linalg.eig(A)
        q = eigenvalues ** 0.5

        Q = np.diag(q)

        X = np.diag(np.exp(-k0*q*d))

        V = W @ Q

        W_i = np.linalg.inv(W)
        V_i = np.linalg.inv(V)

        a = 0.5 * (W_i @ f + V_i @ g)
        b = 0.5 * (W_i @ f - V_i @ g)

        a_i = np.linalg.inv(a)

        f = W @ (np.eye(2*fourier_order+1) + X @ b @ a_i @ X)
        g = V @ (np.eye(2*fourier_order+1) - X @ b @ a_i @ X)
        T = T @ a_i @ X

    Tl = np.linalg.inv(g + 1j*Y_I@f) @ (1j*Y_I@delta_i0 + 1j*n_I*np.cos(theta)*delta_i0)
    R = f @ Tl - delta_i0
    T = T @ Tl

    DEri = R*np.conj(R)*np.real(k_I_z/(k0*n_I*np.cos(theta)))
    DEti = T*np.conj(T)*np.real(k_II_z/(k0*n_I*np.cos(theta)))

    spectrum_r.append(DEri.sum())
    spectrum_t.append(DEti.sum())

plt.plot(wls, spectrum_r)
plt.plot(wls, spectrum_t)
plt.show()
