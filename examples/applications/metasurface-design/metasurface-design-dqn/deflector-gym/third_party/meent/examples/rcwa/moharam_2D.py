import matplotlib.pyplot as plt
import numpy as np

from scipy.linalg import circulant


def to_conv_mat(permittivities, fourier_order):
    # FFT scaling
    # https://kr.mathworks.com/matlabcentral/answers/15770-scaling-the-fft-and-the-ifft#:~:text=the%20matlab%20fft%20outputs%202,point%20is%20the%20parseval%20equation.
    ff = 2 * fourier_order + 1

    if len(permittivities[0].shape) == 1:  # 1D
        res = np.ndarray((len(permittivities), 2*fourier_order+1, 2*fourier_order+1)).astype('complex')

        # extend array
        if permittivities.shape[1] < 2 * ff + 1:
            n = (2 * ff + 1) // permittivities.shape[1]
            permittivities = np.repeat(permittivities, n+1, axis=1)

        for i, pmtvy in enumerate(permittivities):
            pmtvy_fft = np.fft.fftn(pmtvy / pmtvy.size)
            pmtvy_fft = np.fft.fftshift(pmtvy_fft)

            center = len(pmtvy_fft) // 2
            pmtvy_fft_cut = (pmtvy_fft[-ff + center: center+ff+1])
            A = np.roll(circulant(pmtvy_fft_cut.flatten()), (pmtvy_fft_cut.size + 1) // 2, 0)
            res[i] = A[:2*fourier_order+1, :2*fourier_order+1]
            # res[i] = circulant(pmtvy_fft_cut)

    else:  # 2D
        res = np.ndarray((len(permittivities), ff ** 2, ff ** 2)).astype('complex')

        # extend array
        if permittivities.shape[0] < 2 * ff + 1:
            n = (2 * ff + 1) // permittivities.shape[1]
            permittivities = np.repeat(permittivities, n+1, axis=0)
        if permittivities.shape[1] < 2 * ff + 1:
            n = (2 * ff + 1) // permittivities.shape[1]
            permittivities = np.repeat(permittivities, n+1, axis=1)

        for i, pmtvy in enumerate(permittivities):

            pmtvy_fft = np.fft.fftn(pmtvy / pmtvy.size)
            pmtvy_fft = np.fft.fftshift(pmtvy_fft)

            center = np.array(pmtvy_fft.shape) // 2

            conv_idx = np.arange(ff-1, -ff, -1)
            conv_idx = circulant(conv_idx)[ff-1:, :ff]

            conv_i = np.repeat(conv_idx, ff, axis=1)
            conv_i = np.repeat(conv_i, [ff] * ff, axis=0)

            conv_j = np.tile(conv_idx, (ff, ff))
            res[i] = pmtvy_fft[center[0] + conv_i, center[1] + conv_j]

    # import matplotlib.pyplot as plt
    #
    # plt.figure()
    # plt.imshow(abs(res[0]), cmap='jet')
    # plt.colorbar()
    # plt.show()

    return res

pi = np.pi

n_I = 1
n_II = 1

theta = 0.001 * pi / 180
phi = 0 * pi / 180
psi = 0 * pi / 180

fourier_order = 3
ff = 2 * fourier_order + 1
center = ff * ff

period = (0.7, 0.7)

wls = np.linspace(0.5, 2.3, 100)

I = np.eye(ff ** 2)
O = np.zeros((ff**2, ff**2))

spectrum_r, spectrum_t = [], []

thickness = [0.46, 0.66]

# permittivity in grating layer
permt = np.ones((1024, 1024))
permt[300:601, 300:601] = 3.48**2
permt = np.array([permt, permt])

E_conv_all = to_conv_mat(permt, fourier_order)

fourier_indices = np.arange(-fourier_order, fourier_order + 1)

delta_i0 = np.zeros(ff**2).reshape((-1, 1))
delta_i0[ff**2//2, 0] = 1

for wl in wls:
    k0 = 2 * np.pi / wl

    kx_vector = k0 * (n_I*np.sin(theta)*np.cos(phi) - fourier_indices * (wl/period[0])).astype('complex')
    ky_vector = k0 * (n_I*np.sin(theta)*np.sin(phi) - fourier_indices * (wl/period[1])).astype('complex')

    Kx = np.diag(np.tile(kx_vector, ff).flatten()) / k0
    Ky = np.diag(np.tile(ky_vector.reshape((-1, 1)), ff).flatten()) / k0

    k_I_z = (k0**2 * n_I ** 2 - kx_vector**2 - ky_vector.reshape((-1, 1))**2)**0.5
    k_II_z = (k0**2 * n_II ** 2 - kx_vector**2 - ky_vector.reshape((-1, 1))**2)**0.5

    k_I_z = k_I_z.flatten().conjugate()
    k_II_z = k_II_z.flatten().conjugate()

    varphi = np.arctan(ky_vector.reshape((-1, 1))/kx_vector).flatten()

    Y_I = np.diag(k_I_z / k0)
    Y_II = np.diag(k_II_z / k0)

    Z_I = np.diag(k_I_z / (k0 * n_I ** 2))
    Z_II = np.diag(k_II_z / (k0 * n_II ** 2))

    big_F = np.block([[I, O], [O, 1j * Z_II]])
    big_G = np.block([[1j * Y_II, O], [O, I]])

    big_T = np.eye(ff**2*2)

    for E_conv, d in zip(E_conv_all[::-1], thickness[::-1]):

        E_i = np.linalg.inv(E_conv)

        B = Kx @ E_i @ Kx - I
        D = Ky @ E_i @ Ky - I

        S2_from_S = np.block(
            [
                [Ky ** 2 + B @ E_conv, Kx @ (E_i @ Ky @ E_conv - Ky)],
                [Ky @ (E_i @ Kx @ E_conv - Kx), Kx ** 2 + D @ E_conv]
            ])

        eigenvalues, W = np.linalg.eig(S2_from_S)

        q = eigenvalues ** 0.5

        q_1 = q[:center]
        q_2 = q[center:]

        Q = np.diag(q)
        Q_i = np.linalg.inv(Q)
        U1_from_S = np.block(
            [
                [-Kx @ Ky, Kx ** 2 - E_conv],
                [E_conv - Ky ** 2, Ky @ Kx]
            ]
        )
        V = U1_from_S @ W @ Q_i

        W_11 = W[:center, :center]
        W_12 = W[:center, center:]
        W_21 = W[center:, :center]
        W_22 = W[center:, center:]

        V_11 = V[:center, :center]
        V_12 = V[:center, center:]
        V_21 = V[center:, :center]
        V_22 = V[center:, center:]

        X_1 = np.diag(np.exp(-k0*q_1*d))
        X_2 = np.diag(np.exp(-k0*q_2*d))

        F_c = np.diag(np.cos(varphi))
        F_s = np.diag(np.sin(varphi))

        W_ss = F_c @ W_21 - F_s @ W_11
        W_sp = F_c @ W_22 - F_s @ W_12
        W_ps = F_c @ W_11 + F_s @ W_21
        W_pp = F_c @ W_12 + F_s @ W_22

        V_ss = F_c @ V_11 + F_s @ V_21
        V_sp = F_c @ V_12 + F_s @ V_22
        V_ps = F_c @ V_21 - F_s @ V_11
        V_pp = F_c @ V_22 - F_s @ V_12

        big_I = np.eye(2*(len(I)))
        big_X = np.block([[X_1, O], [O, X_2]])
        big_W = np.block([[W_ss, W_sp], [W_ps, W_pp]])
        big_V = np.block([[V_ss, V_sp], [V_ps, V_pp]])

        big_W_i = np.linalg.inv(big_W)
        big_V_i = np.linalg.inv(big_V)

        big_A = 0.5 * (big_W_i @ big_F + big_V_i @ big_G)
        big_B = 0.5 * (big_W_i @ big_F - big_V_i @ big_G)

        big_A_i = np.linalg.inv(big_A)

        big_F = big_W @ (big_I + big_X @ big_B @ big_A_i @ big_X)
        big_G = big_V @ (big_I - big_X @ big_B @ big_A_i @ big_X)

        big_T = big_T @ big_A_i @ big_X

    big_F_11 = big_F[:center, :center]
    big_F_12 = big_F[:center, center:]
    big_F_21 = big_F[center:, :center]
    big_F_22 = big_F[center:, center:]

    big_G_11 = big_G[:center, :center]
    big_G_12 = big_G[:center, center:]
    big_G_21 = big_G[center:, :center]
    big_G_22 = big_G[center:, center:]

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

    R_s = final_X[:ff**2, :].flatten()
    R_p = final_X[ff**2:2*ff**2, :].flatten()

    big_T = big_T @ final_X[2*ff**2:, :]
    T_s = big_T[:ff**2, :].flatten()
    T_p = big_T[ff**2:, :].flatten()

    DEri = R_s*np.conj(R_s) * np.real(k_I_z/(k0*n_I*np.cos(theta))) \
           + R_p*np.conj(R_p) * np.real((k_I_z/n_I**2)/(k0*n_I*np.cos(theta)))

    DEti = T_s*np.conj(T_s) * np.real(k_II_z/(k0*n_I*np.cos(theta))) \
           + T_p*np.conj(T_p) * np.real((k_II_z/n_II**2)/(k0*n_I*np.cos(theta)))

    spectrum_r.append(DEri.sum())
    spectrum_t.append(DEti.sum())

plt.plot(wls, spectrum_r)
plt.plot(wls, spectrum_t)
plt.title(f'Moharam 2D, f order of {fourier_order}')
plt.show()
