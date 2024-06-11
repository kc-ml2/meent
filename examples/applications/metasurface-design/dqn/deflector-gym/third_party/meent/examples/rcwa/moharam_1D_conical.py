import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import circulant


def E_conv_1D_analytic(fourier_order, patterns, period):
    eps = np.zeros(2 * fourier_order + 1).astype('complex')
    E_conv_all = []

    for i, (n_rd, n_gr, fill_factor) in enumerate(patterns):

        for i, order in enumerate(range(-fourier_order, fourier_order + 1)):
            if order == 0: continue
            eps[i] = (n_rd ** 2 - n_gr ** 2) * np.sin(np.pi * order * fill_factor) / (np.pi * order)
            eps[i] *= np.exp(1j * (2 * np.pi * order) / period)

        eps[fourier_order] = (n_rd ** 2 * fill_factor + n_gr ** 2 * (1 - fill_factor))

        E = circulant(np.concatenate((eps[fourier_order:], eps[:fourier_order])))
        E_conv_all.append(E)

    return E_conv_all


pi = np.pi

n_I = 1
n_II = 1

theta = 0.001 * pi / 180
phi = 0 * pi / 180
psi = 0 * pi / 180

fourier_order = 20
ff = 2 * fourier_order + 1

period = 0.7

wls = np.linspace(0.5, 2.3, 400)

I = np.eye(ff)
O = np.zeros((ff, ff))

spectrum_r, spectrum_t = [], []

# permittivity in grating layer
patterns = [[3.48, 1, 0.3], [3.48, 1, 0.3]]  # n_ridge, n_groove, fill_factor
thickness = [0.46, 0.66]
# thickness = [0.46]

E_conv_all = E_conv_1D_analytic(fourier_order, patterns, period)

fourier_indices = np.arange(-fourier_order, fourier_order + 1)

delta_i0 = np.zeros(ff).reshape((-1, 1))
delta_i0[fourier_order] = 1

for wl in wls:
    k0 = 2 * np.pi / wl

    kx_vector = k0*(n_I*np.sin(theta)*np.cos(phi) - fourier_indices * (wl/period)).astype('complex')
    ky = k0 * n_I * np.sin(theta) * np.sin(phi)

    Kx = np.diag(kx_vector/k0)

    k_I_z = (k0**2 * n_I ** 2 - kx_vector**2 - ky**2)**0.5
    k_II_z = (k0**2 * n_II ** 2 - kx_vector**2 - ky**2)**0.5

    k_I_z = k_I_z.conjugate()
    k_II_z = k_II_z.conjugate()

    varphi = np.arctan(ky/kx_vector)

    Y_I = np.diag(k_I_z / k0)
    Y_II = np.diag(k_II_z / k0)

    Z_I = np.diag(k_I_z / (k0 * n_I ** 2))
    Z_II = np.diag(k_II_z / (k0 * n_II ** 2))

    big_F = np.block([[I, O], [O, 1j * Z_II]])
    big_G = np.block([[1j * Y_II, O], [O, I]])

    big_T = np.eye(2*ff)

    for E_conv, d in zip(E_conv_all[::-1], thickness[::-1]):

        E_i = np.linalg.inv(E_conv)

        A = Kx**2 - E_conv
        B = Kx @ E_i @ Kx - I
        A_i = np.linalg.inv(A)
        B_i = np.linalg.inv(B)

        to_decompose_W_1 = ky ** 2 * I + A
        to_decompose_W_2 = ky ** 2 * I + B @ E_conv

        eigenvalues_1, W_1 = np.linalg.eig(to_decompose_W_1)
        eigenvalues_2, W_2 = np.linalg.eig(to_decompose_W_2)

        q_1 = eigenvalues_1 ** 0.5
        q_2 = eigenvalues_2 ** 0.5

        Q_1 = np.diag(q_1)
        Q_2 = np.diag(q_2)

        V_11 = A_i @ W_1 @ Q_1
        V_12 = (ky/k0) * A_i @ Kx @ W_2
        V_21 = (ky/k0) * B_i @ Kx @ E_i @ W_1
        V_22 = B_i @ W_2 @ Q_2

        X_1 = np.diag(np.exp(-k0*q_1*d))
        X_2 = np.diag(np.exp(-k0*q_2*d))

        F_c = np.diag(np.cos(varphi))
        F_s = np.diag(np.sin(varphi))

        V_ss = F_c @ V_11
        V_sp = F_c @ V_12 - F_s @ W_2
        W_ss = F_c @ W_1 + F_s @ V_21
        W_sp = F_s @ V_22
        W_ps = F_s @ V_11
        W_pp = F_c @ W_2 + F_s @ V_12
        V_ps = F_c @ V_21 - F_s @ W_1
        V_pp = F_c @ V_22

        big_I = np.eye(2*(len(I)))
        big_X = np.block([[X_1, O], [O, X_2]])
        big_W = np.block([[V_ss, V_sp], [W_ps, W_pp]])
        big_V = np.block([[W_ss, W_sp], [V_ps, V_pp]])

        big_W_i = np.linalg.inv(big_W)
        big_V_i = np.linalg.inv(big_V)

        big_A = 0.5 * (big_W_i @ big_F + big_V_i @ big_G)
        big_B = 0.5 * (big_W_i @ big_F - big_V_i @ big_G)

        big_A_i = np.linalg.inv(big_A)

        big_F = big_W @ (big_I + big_X @ big_B @ big_A_i @ big_X)
        big_G = big_V @ (big_I - big_X @ big_B @ big_A_i @ big_X)

        big_T = big_T @ big_A_i @ big_X

    big_F_11 = big_F[:ff, :ff]
    big_F_12 = big_F[:ff, ff:]
    big_F_21 = big_F[ff:, :ff]
    big_F_22 = big_F[ff:, ff:]

    big_G_11 = big_G[:ff, :ff]
    big_G_12 = big_G[:ff, ff:]
    big_G_21 = big_G[ff:, :ff]
    big_G_22 = big_G[ff:, ff:]

    # Final Equation in form of AX=B
    final_A = np.block(
        [
            [I, O, -big_F_11, -big_F_12],
            [O, -1j*Z_I, -big_F_21, -big_F_22],
            [-1j*Y_I, O, -big_G_11, -big_G_12],
            [O, I, -big_G_21, -big_G_22],
        ]
    )

    final_B = np.block([
            [-np.sin(psi)*delta_i0],
            [-np.cos(psi) * np.cos(theta) * delta_i0],
            [-1j*np.sin(psi) * n_I * np.cos(theta) * delta_i0],
            [1j*n_I*np.cos(psi) * delta_i0]
        ]
    )

    final_X = np.linalg.inv(final_A) @ final_B

    R_s = final_X[:ff, :].flatten()
    R_p = final_X[ff:2*ff, :].flatten()

    big_T = big_T @ final_X[2*ff:, :]
    T_s = big_T[:ff, :].flatten()
    T_p = big_T[ff:, :].flatten()

    DEri = R_s*np.conj(R_s) * np.real(k_I_z/(k0*n_I*np.cos(theta))) \
           + R_p*np.conj(R_p) * np.real((k_I_z/n_I**2)/(k0*n_I*np.cos(theta)))

    DEti = T_s*np.conj(T_s) * np.real(k_II_z/(k0*n_I*np.cos(theta))) \
           + T_p*np.conj(T_p) * np.real((k_II_z/n_II**2)/(k0*n_I*np.cos(theta)))

    spectrum_r.append(DEri.sum())
    spectrum_t.append(DEti.sum())

plt.plot(wls, spectrum_r)
plt.plot(wls, spectrum_t)
plt.title(f'Moharam 1D conical, f order of {fourier_order}')

plt.show()
