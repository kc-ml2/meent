import numpy as np
import jax
from functools import partial

from jax import tree_util


# @partial(jax.jit, static_argnums=(0, 1))
def call_solver(mode=0, *args, **kwargs):
    """
    decide backend and return RCWA solver instance

    Args:
        mode: decide backend. 0 is numpy and 1 is JAX.
        *args: passed to RCWA instance
        **kwargs: passed to RCWA instance

    Returns:
        RCWA: RCWA solver instance

    """
    if mode == 0:
        from .on_numpy.rcwa import RCWANumpy
        RCWA = RCWANumpy(mode, *args, **kwargs)
    elif mode == 1:
        from .on_jax.rcwa import RCWAJax
        RCWA = RCWAJax(mode=mode, *args, **kwargs)

    elif mode == 2:
        from .on_torch.rcwa import RCWATorch
        RCWA = RCWATorch(mode, *args, **kwargs)
    else:
        raise ValueError

    return RCWA


# def sweep_wavelength(wavelength_array, mode=0, *args, **kwargs):
#     # wavelength = np.linspace(500, 1000, 10)
#     # spectrum_r = []
#     # spectrum_t = []
#     spectrum_r = np.zeros(wavelength_array.shape[0])
#     spectrum_t = np.zeros(wavelength_array.shape[0])
#     solver = call_solver(mode, *args, **kwargs)
#     spectrum_r, spectrum_t = init_spectrum_array(solver.grating_type, wavelength_array, solver.fourier_order)
#
#     for i, wavelength in enumerate(wavelength_array):
#
#         solver.wavelength = np.array([wavelength])
#         de_ri, de_ti = solver.run_ucell()
#         # spectrum_r.append(de_ri)
#         # spectrum_t.append(de_ti)
#         spectrum_r[i] = de_ri
#         spectrum_t[i] = de_ti
#
#     # for i, wavelength in enumerate(wavelength):
#     #     wavelength = np.array([wavelength])
#     #     solver = call_solver(wavelength=wavelength, *args, **kwargs)
#     #     de_ri, de_ti = solver.run_ucell()
#     #     spectrum_r.append(de_ri)
#     #     spectrum_t.append(de_ti)
#     #     # spectrum_r[i] = de_ri
#     #     # spectrum_t[i] = de_ti
#
#     # spectrum_r = np.array(spectrum_r)
#     # spectrum_t = np.array(spectrum_t)
#
#     return spectrum_r, spectrum_t
#
#
# def init_spectrum_array(grating_type, wavelength_array, fourier_order):
#     if grating_type in (0, 1):
#         spectrum_r = np.zeros((len(wavelength_array), 2 * fourier_order + 1))
#         spectrum_t = np.zeros((len(wavelength_array), 2 * fourier_order + 1))
#     elif grating_type == 2:
#         spectrum_r = np.zeros((len(wavelength_array), 2 * fourier_order + 1, 2 * fourier_order + 1))
#         spectrum_t = np.zeros((len(wavelength_array), 2 * fourier_order + 1, 2 * fourier_order + 1))
#     else:
#         raise ValueError
#     return spectrum_r, spectrum_t
