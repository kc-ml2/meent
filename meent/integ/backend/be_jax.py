
import numpy

import jax
import jax.numpy as np
from jax import jit

backend = 'jax'


linspace = np.linspace
pi = np.pi
arange = np.arange
zeros = np.zeros
diag = np.diag

linalg = np.linalg
eye = np.eye
nan = np.nan
interp = np.interp
loadtxt = numpy.loadtxt
array = np.array
roll = np.roll
exp = np.exp
vstack = np.vstack
ones = np.ones
repeat = np.repeat
tile = np.tile

sin = np.sin
cos = np.cos
block = np.block

real = np.real
imag = np.imag
conj = np.conj

nonzero = np.nonzero

arctan = np.arctan

hstack = np.hstack
eig = np.linalg.eig

from jax import jit


# @jit
def _assign_row_all(arr, index, value):
    row, col = arr.shape
    return arr.at[:, index].set(value)


@jit
def _assign_col_all(arr, index, value):
    row, col = arr.shape
    return arr.at[index, np.arange(col)].set(value)


@jit
def _assign(arr, index, value):
    return arr.at[index].set(value)


def assign(arr, index, value, row_all=False, col_all=False):
    if type(index) == list:
        index = tuple(index)

    if row_all:
        arr = _assign_row_all(arr, index, value)
    elif col_all:
        arr = _assign_col_all(arr, index, value)
    else:
        arr = _assign(arr, index, value)
    return arr


def assign1(arr, index, value, row_all=False, col_all=False):
    if type(index) == list:
        index = tuple(index)

    if row_all:
        arr = arr.at[:, index].set(value)
    elif col_all:
        arr = arr.at[index, :].set(value)
    else:
        arr = arr.at[index].set(value)
    return arr



# import numpy
#
# import jax
# import jax.numpy as np
# from jax import jit
#
# backend = 'jax'

#
# linspace = jit(np.linspace)
# pi = np.pi
# arange = jit(np.arange)
# zeros = np.zeros
# diag = jit(np.diag)
#
# linalg = np.linalg
# eye = jit(np.eye)
# nan = np.nan
# interp = jit(np.interp)
# loadtxt = numpy.loadtxt
# array = jit(np.array)
# roll = jit(np.roll)
# exp = jit(np.exp)
# vstack = jit(np.vstack)
# ones = jit(np.ones)
# repeat = jit(np.repeat)
# tile = jit(np.tile)
#
# sin = jit(np.sin)
# cos = jit(np.cos)
# block = jit(np.block)
#
# real = jit(np.real)
# imag = jit(np.imag)
# conj = jit(np.conj)
#
# nonzero = jit(np.nonzero)
#
# arctan = jit(np.arctan)
#
# hstack = jit(np.hstack)
# eig = np.linalg.eig
#
#
# def assign(arr, index, value, row_all=False, col_all=False):
#     if type(index) == list:
#         index = tuple(index)
#
#     if row_all:
#         arr = arr.at[:, index].set(value)
#     elif col_all:
#         arr = arr.at[index, :].set(value)
#     else:
#         arr = arr.at[index].set(value)
#     return arr


