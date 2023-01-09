import meent.integ.backend

if meent.integ.backend.mode == 2:
    from meent.integ.backend.be_numpy import *
    print(33)
    # import numpy as np
    #
    # backend = 'numpy'
    # linspace = np.linspace
    # pi = np.pi
    # arange = np.arange
    # zeros = np.zeros
    # diag = np.diag
    #
    # linalg = np.linalg
    # eye = np.eye
    # nan = np.nan
    # interp = np.interp
    # loadtxt = np.loadtxt
    # array = np.array
    # roll = np.roll
    # exp = np.exp
    # vstack = np.vstack
    # ones = np.ones
    # repeat = np.repeat
    # tile = np.tile
    #
    # sin = np.sin
    # cos = np.cos
    # block = np.block
    #
    # real = np.real
    # imag = np.imag
    # conj = np.conj
    #
    # nonzero = np.nonzero
    #
    # arctan = np.arctan
    #
    # hstack = np.hstack
    #
    # eig = np.linalg.eig




elif meent.integ.backend.mode == 3:
    from meent.integ.backend.be_jax import *
    print(23)
