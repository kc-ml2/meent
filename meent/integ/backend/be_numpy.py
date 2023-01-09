import numpy as np


backend = 'numpy'
linspace = np.linspace
pi = np.pi
arange = np.arange
zeros = np.zeros
diag = np.diag

linalg = np.linalg
eye = np.eye
nan = np.nan
interp = np.interp
loadtxt = np.loadtxt
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
