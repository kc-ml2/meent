import jax
import jax.numpy as jnp

from meent.on_jax.convolution_matrix import to_conv_mat, put_permittivity_in_ucell, read_material_table
from meent.on_jax.transfer_method import *
from ex_ucell import load_ucell


@partial(jax.jit, static_argnums=(1, 2, 7, ))
def get_kx_vector(wavelength, fourier_order, grating_type, n_I, theta, phi, period, type_complex,
                  perturbation):

    k0 = 2 * jnp.pi / wavelength
    fourier_indices = jnp.arange(-fourier_order, fourier_order + 1)
    if grating_type == 0:
        kx_vector = k0 * (n_I * jnp.sin(theta) - fourier_indices * (wavelength / period[0])
                          ).astype(type_complex)
    else:
        kx_vector = k0 * (n_I * jnp.sin(theta) * jnp.cos(phi) - fourier_indices * (
                wavelength / period[0])).astype(type_complex)


    kx_vector = jnp.where(kx_vector == 0, perturbation, kx_vector)
    # TODO: need imaginary part? make imaginary part sign consistent
    # TODO: perturbation added silently

    return kx_vector



@partial(jax.jit, static_argnums=(3, 12, ))
def solve_2d(wavelength, E_conv_all, o_E_conv_all, fourier_order, n_I, n_II, kx_vector, period,
             theta, phi, psi, thickness, type_complex):
    layer_info_list = []
    T1 = None
    ff = 2 * fourier_order + 1

    fourier_indices = jnp.arange(-fourier_order, fourier_order + 1)

    delta_i0 = jnp.zeros((ff ** 2, 1), dtype=type_complex)
    delta_i0 = delta_i0.at[ff ** 2 // 2, 0].set(1)

    I = jnp.eye(ff ** 2).astype(type_complex)
    O = jnp.zeros((ff ** 2, ff ** 2), dtype=type_complex)

    center = ff ** 2

    k0 = 2 * jnp.pi / wavelength

    kx_vector, ky_vector, Kx, Ky, k_I_z, k_II_z, varphi, Y_I, Y_II, Z_I, Z_II, big_F, big_G, big_T \
        = transfer_2d_1(ff, k0, n_I, n_II, kx_vector, period, fourier_indices,
                        theta, phi, wavelength, type_complex=type_complex)

    # From the last layer
    for E_conv, o_E_conv, d in zip(E_conv_all[::-1], o_E_conv_all[::-1], thickness[::-1]):
        E_conv_i = jnp.linalg.inv(E_conv)
        o_E_conv_i = jnp.linalg.inv(o_E_conv)

        W, V, q = transfer_2d_wv(ff, Kx, E_conv_i, Ky, o_E_conv_i, E_conv, type_complex=type_complex)

        big_X, big_F, big_G, big_T, big_A_i, big_B, \
        W_11, W_12, W_21, W_22, V_11, V_12, V_21, V_22 \
            = transfer_2d_2(k0, d, W, V, center, q, varphi, I, O, big_F, big_G, big_T,
                            type_complex=type_complex)

        layer_info = [E_conv_i, q, W_11, W_12, W_21, W_22, V_11, V_12, V_21, V_22, big_X, big_A_i, big_B, d]
        layer_info_list.append(layer_info)

    de_ri, de_ti, big_T1 = transfer_2d_3(center, big_F, big_G, big_T, Z_I, Y_I, psi, theta, ff,
                                         delta_i0, k_I_z, k0, n_I, n_II, k_II_z,
                                         type_complex=type_complex)
    T1 = big_T1

    return de_ri.reshape((ff, ff)).real, de_ti.reshape((ff, ff)).real, layer_info_list, T1



ucell = load_ucell(2)
mat_list = [1, 3.48]
wavelength = 900
fourier_order = 2
type_complex = jnp.complex64

with jax.default_device(jax.devices("cpu")[0]):
    mat_table = read_material_table(type_complex=type_complex)
    ucell = put_permittivity_in_ucell(ucell, mat_list, mat_table, wavelength, type_complex=type_complex)

    E_conv_all = to_conv_mat(ucell, fourier_order, type_complex=type_complex)
    o_E_conv_all = to_conv_mat(1 / ucell, fourier_order, type_complex=type_complex)

    grating_type = 2
    n_I = 1
    n_II = 1
    period = [10, 10]
    theta = 0
    phi = 0
    thickness = [10]
    psi = 0
    perturbation = 1E-10

    kx_vector = get_kx_vector(wavelength, fourier_order, grating_type, n_I, theta, phi, period, type_complex, perturbation)

    de_ri, de_ti, layer_info_list, T1 = solve_2d(wavelength, E_conv_all, o_E_conv_all, fourier_order, n_I, n_II,
                                                 kx_vector, period,
                                                 theta, phi, psi, thickness, type_complex)

print(1)
