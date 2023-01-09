import numpy as np

import jax
import jax.numpy as jnp
from functools import partial

from jax.experimental import host_callback

loadtxt = np.loadtxt


backend = 'jax'

pi = jnp.pi

diag = jax.jit(jnp.diag)


inv = jax.jit(jnp.linalg.inv)

nan = jnp.nan
interp = jax.jit(jnp.interp)

exp = jax.jit(jnp.exp)
vstack = jax.jit(jnp.vstack)

sin = jax.jit(jnp.sin)
cos = jax.jit(jnp.cos)
block = jax.jit(jnp.block)

real = jax.jit(jnp.real)
imag = jax.jit(jnp.imag)
conj = jax.jit(jnp.conj)


arctan = jax.jit(jnp.arctan)

hstack = jax.jit(jnp.hstack)


array = partial(jax.jit, static_argnums=(1, ))(jnp.array)

roll = partial(jax.jit, static_argnums=(2,))(jnp.roll)
arange = partial(jax.jit, static_argnums=(0, 1, 2))(jnp.arange)

ones = partial(jax.jit, static_argnums=(0, 1))(jnp.ones)
zeros = partial(jax.jit, static_argnums=(0, 1))(jnp.zeros)

repeat = partial(jax.jit, static_argnums=(1, 2, ))(jnp.repeat)
tile = partial(jax.jit, static_argnums=(1, ))(jnp.tile)

linspace = partial(jax.jit, static_argnums=(0, 1, 2))(jnp.linspace)

eye = partial(jax.jit, static_argnums=(0, ))(jnp.eye)
nonzero = partial(jax.jit, static_argnums=(0, ))(jnp.nonzero)


@partial(jax.jit, static_argnums=(3, 4))
def assign(arr, index, value, row_all=False, col_all=False):
    if type(index) == list:
        index = tuple(index)

    if row_all:

        print('assign_new')
        # # coord = jnp.array([[r,c] for c in index for r in range(arr.shape[0])]).T
        # coord1 = jnp.array([[[r, c] for c in index] for r in range(arr.shape[0])])  # TODO: remove loop
        # coord = tuple(jnp.moveaxis(coord1, -1, 0))
        #
        # arr = arr.at[coord].set(value)
        arr = arr.at[:, index].set(value)
    elif col_all:
        arr = arr.at[index, :].set(value)
    else:
        arr = arr.at[index].set(value)
    return arr


@partial(jax.jit, static_argnums=(1, ))
def eig(matrix: jnp.ndarray, type_complex=jnp.complex128) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Wraps jnp.linalg.eig so that it can be jit-ed on a machine with GPUs."""
    eigenvalues_shape = jax.ShapeDtypeStruct(matrix.shape[:-1], type_complex)
    eigenvectors_shape = jax.ShapeDtypeStruct(matrix.shape, type_complex)
    return host_callback.call(
        # We force this computation to be performed on the cpu by jit-ing and
        # explicitly specifying the device.
        jax.jit(jnp.linalg.eig, device=jax.devices('cpu')[0]),
        matrix.astype(type_complex),
        result_shape=(eigenvalues_shape, eigenvectors_shape),
    )
