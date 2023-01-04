import numpy as np

import jax
import jax.numpy as jnp
from functools import partial

from jax.experimental import host_callback

loadtxt = np.loadtxt


backend = 'jax'

pi = jnp.pi
diag = jnp.diag


inv = jnp.linalg.inv

nan = jnp.nan
interp = jnp.interp
array = jnp.array

exp = jnp.exp
vstack = jnp.vstack

sin = jnp.sin
cos = jnp.cos
block = jnp.block

real = jnp.real
imag = jnp.imag
conj = jnp.conj


arctan = jnp.arctan

hstack = jnp.hstack

roll =jnp.roll
arange = jnp.arange

ones = jnp.ones
zeros = jnp.zeros

repeat = jnp.repeat
tile = jnp.tile
linspace =jnp.linspace
eye = jnp.eye
nonzero = jnp.nonzero



# roll = partial(jax.jit, static_argnums=(2,))(jnp.roll)
# arange = partial(jax.jit, static_argnums=(0, 1, 2))(jnp.arange)
#
# ones = partial(jax.jit, static_argnums=(0, 1))(jnp.ones)
# zeros = partial(jax.jit, static_argnums=(0, 1))(jnp.zeros)
#
# repeat = partial(jax.jit, static_argnums=(1, 2, ))(jnp.repeat)  # TODO: exclude 1 from static
# tile = partial(jax.jit, static_argnums=(1, ))(jnp.tile)  # TODO: exclude 1 from static
#
# linspace = partial(jax.jit, static_argnums=(0, 1, 2))(jnp.linspace)
#
# eye = partial(jax.jit, static_argnums=(0, ))(jnp.eye)
#
#
# nonzero = partial(jax.jit, static_argnums=(0, ))(jnp.nonzero)

@partial(jax.jit, static_argnums=(3, 4))
def assign(arr, index, value, row_all=False, col_all=False):
    # print('assign_new')
    if type(index) == list:
        index = tuple(index)

    if row_all:

        # coord = jnp.array([[r,c] for c in index for r in range(arr.shape[0])]).T
        coord1 = jnp.array([[[r, c] for c in index] for r in range(arr.shape[0])])
        coord = tuple(jnp.moveaxis(coord1, -1, 0))

        # coord = tuple(jnp.array([[[r, c] for c in index] for r in range(arr.shape[0])]).T)

        arr = arr.at[coord].set(value)
        # arr = arr.at[:, index].set(value)
    elif col_all:
        arr = arr.at[index, :].set(value)
    else:
        arr = arr.at[index].set(value)
    return arr


# eig = jax.jit(jnp.linalg.eig)
def _eig_host(matrix: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Wraps jnp.linalg.eig so that it can be jit-ed on a machine with GPUs."""
    print('_eig_host_compile')
    eigenvalues_shape = jax.ShapeDtypeStruct(matrix.shape[:-1], complex)
    eigenvectors_shape = jax.ShapeDtypeStruct(matrix.shape, complex)
    return host_callback.call(
        # We force this computation to be performed on the cpu by jit-ing and
        # explicitly specifying the device.
        jax.jit(jnp.linalg.eig, device=jax.devices("cpu")[0]),
        matrix.astype(complex),
        result_shape=(eigenvalues_shape, eigenvectors_shape),
    )


eig = jax.jit(_eig_host)  # This works, we can jit on GPU.
# eig = jax.jit(_eig_host, device=jax.devices("cpu")[0])  # This works, we can jit on GPU.
# eig = jnp.linalg.eig


# import time
# import numpy as np
#
# import jax
# import jax.numpy as jnp
# from functools import partial
#
# from jax.experimental import host_callback
#
# loadtxt = np.loadtxt
#
#
# backend = 'jax'
#
# pi = jnp.pi
# diag = jax.jit(jnp.diag)
#
#
# inv = jax.jit(jnp.linalg.inv)
#
# nan = jnp.nan
# interp = jax.jit(jnp.interp)
# array = jax.jit(jnp.array)
#
# exp = jax.jit(jnp.exp)
# vstack = jax.jit(jnp.vstack)
#
# sin = jax.jit(jnp.sin)
# cos = jax.jit(jnp.cos)
# block = jax.jit(jnp.block)
#
# real = jax.jit(jnp.real)
# imag = jax.jit(jnp.imag)
# conj = jax.jit(jnp.conj)
#
#
# arctan = jax.jit(jnp.arctan)
#
# hstack = jax.jit(jnp.hstack)
#
#
#
# roll = partial(jax.jit, static_argnums=(2,))(jnp.roll)
# arange = partial(jax.jit, static_argnums=(0, 1, 2))(jnp.arange)
#
# ones = partial(jax.jit, static_argnums=(0, 1))(jnp.ones)
# zeros = partial(jax.jit, static_argnums=(0, 1))(jnp.zeros)
#
# repeat = partial(jax.jit, static_argnums=(1, 2, ))(jnp.repeat)  # TODO: exclude 1 from static
# tile = partial(jax.jit, static_argnums=(1, ))(jnp.tile)  # TODO: exclude 1 from static
#
# linspace = partial(jax.jit, static_argnums=(0, 1, 2))(jnp.linspace)
#
# eye = partial(jax.jit, static_argnums=(0, ))(jnp.eye)
#
#
# nonzero = partial(jax.jit, static_argnums=(0, ))(jnp.nonzero)
#
# @partial(jax.jit, static_argnums=(3, 4))
# def assign(arr, index, value, row_all=False, col_all=False):  # TODO: make individual assign function per condition
#     # print('assign_new')
#     # t0 = time.time()
#     if type(index) == list:
#         index = tuple(index)
#
#     if row_all:
#         # print('assign_row_all')
#         coord = jnp.array([[[r, c] for c in index] for r in range(arr.shape[0])])  # TODO: improve
#         # print(time.time()-t0)
#         coord = tuple(jnp.moveaxis(coord, -1, 0))
#
#         # t1 = time.time()
#         # print('assign_row_all_prep_done', t1 - t0)
#         # t2 = time.time()
#
#         arr = arr.at[coord].set(value)
#         # print('assign_row_all_done', time.time() - t2)
#
#     elif col_all:
#         raise ValueError('assign for col_all is not implemented')
#         # arr = arr.at[index, :].set(value)
#     else:
#         arr = arr.at[index].set(value)
#         # print('assign_base_done', time.time()-t0)
#
#     return arr
#
#
# # eig = jax.jit(jnp.linalg.eig)
# def _eig_host(matrix: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
#     """Wraps jnp.linalg.eig so that it can be jit-ed on a machine with GPUs."""
#     print('_eig_host_compile')
#     eigenvalues_shape = jax.ShapeDtypeStruct(matrix.shape[:-1], complex)
#     eigenvectors_shape = jax.ShapeDtypeStruct(matrix.shape, complex)
#     return host_callback.call(
#         # We force this computation to be performed on the cpu by jit-ing and
#         # explicitly specifying the device.
#         jax.jit(jnp.linalg.eig, device=jax.devices("cpu")[0]),
#         matrix.astype(complex),
#         result_shape=(eigenvalues_shape, eigenvectors_shape),
#     )
#
#
# # eig = jax.jit(_eig_host, device=jax.devices("gpu")[0])  # This works, we can jit on GPU.
# # eig = jax.jit(_eig_host, device=jax.devices("cpu")[0])  # This works, we can jit on GPU.
# eig = jax.jit(_eig_host)  # This works, we can jit on GPU.
#
