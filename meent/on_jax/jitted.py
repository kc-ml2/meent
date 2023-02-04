import numpy as np

import jax
import jax.numpy as jnp
from functools import partial

from jax.experimental import host_callback
from jax import pure_callback

loadtxt = np.loadtxt

backend = 'jax'

pi = jnp.pi

diag = jax.jit(jnp.diag)

inv = jax.jit(jnp.linalg.inv)

nan = jnp.nan

# TODO
# interp = jax.jit(jnp.interp)
interp = jnp.interp

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

array = partial(jax.jit, static_argnums=(1,))(jnp.array)

roll = partial(jax.jit, static_argnums=(2,))(jnp.roll)
arange = partial(jax.jit, static_argnums=(0, 1, 2))(jnp.arange)

ones = partial(jax.jit, static_argnums=(0, 1))(jnp.ones)
zeros = partial(jax.jit, static_argnums=(0, 1))(jnp.zeros)

repeat = partial(jax.jit, static_argnums=(1, 2,))(jnp.repeat)
tile = partial(jax.jit, static_argnums=(1,))(jnp.tile)

linspace = partial(jax.jit, static_argnums=(0, 1, 2))(jnp.linspace)

eye = partial(jax.jit, static_argnums=(0,))(jnp.eye)
nonzero = partial(jax.jit, static_argnums=(0,))(jnp.nonzero)


@partial(jax.jit, static_argnums=(3, 4))
def assign(arr, index, value, row_all=False, col_all=False):
    if type(index) == list:
        index = tuple(index)

    if row_all:
        arr = arr.at[:, index].set(value)
    elif col_all:
        arr = arr.at[index, :].set(value)
    else:
        arr = arr.at[index].set(value)
    return arr


def eig2(matrix, type_complex=jnp.complex128):
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


def _eig(X, type_complex):
    import numpy as np
    # jax.config.update('jax_enable_x64', True)

    # type_complex = jnp.complex128
    # X = np.array(X, dtype=np.complex64)
    _eig = lambda x: jax.jit(jnp.linalg.eig, device=jax.devices('cpu')[0])(x)

    eigenvalues_shape = jax.ShapeDtypeStruct(X.shape[:-1], type_complex)
    eigenvectors_shape = jax.ShapeDtypeStruct(X.shape, type_complex)

    result_shape_dtype = (eigenvalues_shape, eigenvectors_shape)

    return jax.pure_callback(_eig, result_shape_dtype, X)


# eig = jnp.linalg.eig

# @partial(jax.jit, backend='cpu')
# def eig2(mat):
#     print('eig')
#     return jnp.linalg.eig(mat)

# @partial(jax.jit, static_argnums=(1, ), backend='cpu')
# def eig(mat, _):
#     # _ in args is used for not to jit compile. Refer https://github.com/google/jax/issues/13959#issue-1528545365
#     return jnp.linalg.eig(mat)


# def eig(mat, _):
#     with jax.default_device(jax.devices('cpu')[0]):
#         a=jnp.linalg.eig(mat)
#         return a

fft = jnp.fft
