import scipy
import numpy as np

from scipy.linalg import expm


def field_dist(f, x, y, z, T, Q, T1):

    X = np.diag(np.exp(-k0 * q * d))

    W_i = np.linalg.inv(W)
    V_i = np.linalg.inv(V)

    a = 0.5 * (W_i @ f1 + V_i @ g1)
    b = 0.5 * (W_i @ f1 - V_i @ g1)

    a_i = np.linalg.inv(a)

    c1 = T1[:, None]
    c2 = b @ a_i @ X @ T1[:, None]

    step = self.period[0] / len_x

    x *= step

    Uy = W @ (scipy.linalg.expm(-k0 * Q * z) @ c1 + scipy.linalg.expm(k0 * Q * (z - d)) @ c2)
    Hy = Uy.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x)

    # Original TMM method (without enhanced)
    # Z_II = np.diag(k_II_z / (k0 * self.n_II ** 2))
    # CC = np.linalg.inv(np.block([[W@X, W], [V@X, -V]])) @ np.block([[np.eye(self.ff)], [1j*Z_II]]) @ T
    # cc1 = CC[:self.ff]
    # cc2 = CC[self.ff:]
    # Uy1 = W @ (scipy.linalg.expm(-k0 * Q * z) @ cc1 + scipy.linalg.expm(k0 * Q * (z - d)) @ cc2)
    # Hy1 = Uy1.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x)


    omega = 2 * np.pi / wl
    G = (1j * k0 / omega) * W @ Q @ (-scipy.linalg.expm(-k0 * Q * z) @ c1 + scipy.linalg.expm(k0 * Q * (z - d)) @ c2)
    Dx = G.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x)
    Ex = Dx if x < 300 else Dx /3.48**2

    # eps0 = 8.854E-12/1E-9
    eps0 = 100 * np.sqrt(2)
    eps0= 1/ omega
    f_here = (-1j / omega / eps0) * np.linalg.inv(E_conv) @ Kx @ Uy
    Ez = f_here.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x)

    return Hy, Ex, Ez


def field_dist_1d_tm(x, y, z, T1, q, k0, d, W, b, a_i, kx_vector, wavelength, E_conv_i, Kx):

    X = np.diag(np.exp(-k0 * q * d))
    Q = np.diag(q)

    c1 = T1[:, None]
    c2 = b @ a_i @ X @ T1[:, None]

    Uy = W @ (expm(-k0 * Q * z) @ c1 + expm(k0 * Q * (z - d)) @ c2)
    Hy = Uy.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x)

    # Original TMM method (without enhanced)
    # Z_II = np.diag(k_II_z / (k0 * self.n_II ** 2))
    # CC = np.linalg.inv(np.block([[W@X, W], [V@X, -V]])) @ np.block([[np.eye(self.ff)], [1j*Z_II]]) @ T
    # cc1 = CC[:self.ff]
    # cc2 = CC[self.ff:]
    # Uy1 = W @ (scipy.linalg.expm(-k0 * Q * z) @ cc1 + scipy.linalg.expm(k0 * Q * (z - d)) @ cc2)
    # Hy1 = Uy1.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x)

    omega = 2 * np.pi / wavelength
    G = (1j * k0 / omega) * W @ Q @ (-expm(-k0 * Q * z) @ c1 + expm(k0 * Q * (z - d)) @ c2)
    Dx = G.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x)
    Ex = Dx if x < 300 else Dx /3.48 **2

    # eps0 = 8.854E-12/1E-9
    eps0 = 1 / omega
    f_here = (-1j / omega / eps0) * E_conv_i @ Kx @ Uy
    Ez = f_here.T @ np.exp(-1j * kx_vector.reshape((-1, 1)) * x)

    return Hy, Ex, Ez
