import tensorflow as tf
import numpy as np


backend = 'numpy'
linspace = tf.linspace
pi = tf.constant(np.pi)

arange = tf.experimental.numpy.arange

# zeros = tf.zeros
zeros = tf.experimental.numpy.zeros

# diag = tf.linalg.diag
diag = tf.experimental.numpy.diag


linalg = tf.linalg
eye = tf.eye

# nan = tf.nan
nan = tf.constant(np.nan)

interp = None

loadtxt = np.loadtxt

array = tf.experimental.numpy.array

# roll = tf.roll
roll = tf.experimental.numpy.roll


exp = tf.exp

# vstack = tf.vstack
vstack = tf.experimental.numpy.vstack


ones = tf.ones
repeat = tf.repeat
tile = tf.tile

sin = tf.sin
cos = tf.cos
block = None

real = tf.math.real
imag = tf.math.imag
conj = tf.math.conj

nonzero = tf.experimental.numpy.nonzero

# arctan = tf.math.atan
arctan = tf.experimental.numpy.arctan

hstack = tf.experimental.numpy.hstack

eig = tf.linalg.eig


def assign(arr, index, value, row_all=False, col_all=False):
    if type(index) == list:
        index = tuple(index)

    if row_all:
        arr[:, index] = value
    elif col_all:
        arr[index, :] = value
    else:
        arr[index] = value
    return arr

# https://stackoverflow.com/questions/38420288/how-to-implement-element-wise-1d-interpolation-in-tensorflow
def interp():
    pass