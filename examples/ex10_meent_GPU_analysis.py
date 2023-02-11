import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'

import jax

# # common
# # grating_type = 1  # 0: 1D, 1: 1D conical, 2:2D.
# pol = 1  # 0: TE, 1: TM
#
# n_I = 1  # n_incidence
# n_II = 1  # n_transmission
#
# theta = 1E-10
# phi = 0
# psi = 0 if pol else 90
#
# wavelength = 900
#
# thickness = [500]
# ucell_materials = [1, 3.48]
#
# mode_options = {0: 'numpy', 1: 'JAX', 2: 'Torch', 3: 'numpy_integ', 4: 'JAX_integ',}
# n_iter = 2

from examples.ex_ucell import run_test, run_loop

# def run_test(grating_type, mode_key, dtype, device):
#
#     ucell = load_ucell(grating_type)
#     if grating_type in [0, 1]:
#         period = [1000]
#         fourier_order = 10
#     else:
#         period = [1000, 1000]
#         fourier_order = 10
#
#     if mode_key == 0:
#         if device != 0:
#             return
#         if dtype == 0:
#             type_complex = np.complex128
#         else:
#             type_complex = np.complex64
#
#     # JAX
#     elif mode_key == 1:
#         if device == 0:
#             device = 'cpu'
#             jax.config.update('jax_platform_name', device)
#         else:
#             device = 'gpu'
#             jax.config.update('jax_platform_name', device)
#
#         if dtype == 0:
#             jax.config.update("jax_enable_x64", True)
#             type_complex = jnp.complex128
#         else:
#             jax.config.update("jax_enable_x64", False)
#             type_complex = jnp.complex64
#
#     else:
#         # Torch
#         if device == 0:
#             device = torch.device('cpu')
#         else:
#             device = torch.device('cuda')
#
#         if dtype == 0:
#             type_complex = torch.complex128
#         else:
#             type_complex = torch.complex64
#
#     AA = call_solver(mode=mode_key, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi, psi=psi,
#                      fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell,
#                      ucell_materials=ucell_materials,
#                      thickness=thickness, device=device, type_complex=type_complex, )
#
#     for i in range(n_iter):
#         t0 = time.time()
#         de_ri, de_ti = AA.run_ucell()
#         print(f'run_cell time {i}: ', np.round(time.time()-t0, 1))
#
#     return de_ri, de_ti


# def run_loop(a,b, c, d):
#     for grating_type in a:
#         for bd in b:
#             for dtype in c:
#                 for device in d:
#                     # run_test(grating_type, bd, dtype, device)
#
#                     try:
#                         print(f'grating:{grating_type}, backend:{bd}, dtype:{dtype}, dev:{device}')
#                         de_ri, de_ti = run_test(grating_type, bd, dtype, device)
#                         # print(de_ti.sum())
#
#                     except Exception as e:
#                         print(e)

a = [2]
b = [1]
c = [0,1]

with jax.default_device(jax.devices("cpu")[0]):
    run_loop(a, b, c, [0])

with jax.default_device(jax.devices("gpu")[0]):
    run_loop(a, b, c, [1])

with jax.default_device(jax.devices("cpu")[0]):
    run_loop(a, b, c, [0])





