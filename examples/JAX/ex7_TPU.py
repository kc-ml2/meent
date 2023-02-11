# import jax
# import jax.numpy as jnp
#
# # jax.device_count()
#
# import meent
#
# import time
#
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
#
# from ex_ucell import load_ucell
#
#
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
# mode_options = {0: 'numpy', 1: 'JAX', 2: 'Torch', }
# n_iter = 1
#
#
# def run_test(grating_type, mode_key, dtype, device):
#     ucell = load_ucell(grating_type)
#     if grating_type in [0, 1]:
#         period = [1000]
#         fourier_order = 10
#     else:
#         period = [1000, 1000]
#         fourier_order = 2
#
#     # Numpy
#     if mode_key == 0:
#         if device != 0:
#             raise ValueError
#         if dtype == 0:
#             type_complex = np.complex128
#         elif dtype == 1:
#             type_complex = np.complex64
#         else:
#             raise ValueError
#
#     # JAX
#     elif mode_key == 1:
#         if device == 0:
#             device = 'cpu'
#             jax.config.update('jax_platform_name', device)
#         elif device == 1:
#             device = 'gpu'
#             jax.config.update('jax_platform_name', 'gpu')
#         elif device == 2:
#             # 2023.02, TPU is not available because of Eigen-decomposition
#             raise ValueError
#
#         if dtype == 0:
#             jax.config.update("jax_enable_x64", True)
#             type_complex = jnp.complex128
#
#         elif dtype == 1:
#             jax.config.update("jax_enable_x64", False)
#             type_complex = jnp.complex64
#         else:
#             raise ValueError
#
#     else:
#         # Torch
#         if device == 0:
#             device = torch.device('cpu')
#         elif device == 1:
#             device = torch.device('cuda')
#         else:
#             raise ValueError
#
#         if dtype == 0:
#             type_complex = torch.complex128
#         elif dtype == 1:
#             type_complex = torch.complex64
#         else:
#             raise ValueError
#
#     AA = meent.call_solver(mode=mode_key, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi,
#                      psi=psi,
#                      fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell,
#                      ucell_materials=ucell_materials,
#                      thickness=thickness, device=device, type_complex=type_complex, fft_type='piecewise')
#
#     for i in range(n_iter):
#         t0 = time.time()
#         de_ri, de_ti = AA.run_ucell()
#         print(f'run_cell: {i}: ', time.time() - t0)
#
#     resolution = (20, 20, 20)
#     for i in range(1):
#         t0 = time.time()
#         AA.calculate_field(resolution=resolution, plot=False)
#         print(f'cal_field: {i}', time.time() - t0)
#
#     return de_ri, de_ti
#
#
# def run_loop(a,b, c, d):
#     for grating_type in a:
#         for bd in b:
#             for dtype in c:
#                 for device in d:
#                     print(f'grating:{grating_type}, backend:{bd}, dtype:{dtype}, dev:{device}')
#                     run_test(grating_type, bd, dtype, device)
#                     # try:
#                     #     print(f'grating:{grating_type}, backend:{bd}, dtype:{dtype}, dev:{device}')
#                     #     run_test(grating_type, bd, dtype, device)
#                     #
#                     # except Exception as e:
#                     #     print(e)
#
# a = [2]
# b = [0]
# c = [0]
#
# with jax.default_device(jax.devices("cpu")[0]):
#     run_loop(a, b, c, [0])
#
# # with jax.default_device(jax.devices("gpu")[0]):
# #     run_loop(a, b, c, [1])
# #
# # with jax.default_device(jax.devices("cpu")[0]):
# #     run_loop(a, b, c, [0])
#
#
#
#
#