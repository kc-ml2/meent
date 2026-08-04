"""1D diagonally-anisotropic grating on a slab, conical incidence.

Validated against RETICOLO (MATLAB). See README.md in this directory for
provenance, the measured agreement, and the caveats.

This module is the machine-readable half of that README: it holds everything
`tests/test_reference.py` needs to rebuild the experiment in meent. It must stay
importable with numpy alone -- no meent, torch or jax at import time.
"""

import numpy as np

NAME = '1D_anisotropic_grating_conical_incidence'
DESCRIPTION = (
    'Diagonally anisotropic (nx, ny, nz) = (1.5, 2.0, 2.5) grating bar on a slab '
    'of the same tensor, conical incidence (theta = 30 deg, phi = 30 deg), '
    '500-700 nm. Reference: RETICOLO.'
)

# What a checkout must be able to do before the live tests mean anything.
# 'anisotropy_diagonal' is probed by _loader._probe_anisotropy_diagonal; it
# landed on feature/anisotropic-diagonal-torch and is torch-only.
REQUIRES = {
    'backends': [2],
    'features': ['anisotropy_diagonal'],
}

# --------------------------------------------------------------------------- #
# structure -- SI units (metres), exactly as the notebook that produced recorded/
# --------------------------------------------------------------------------- #

BACKEND = 2               # torch
TYPE_COMPLEX = 0          # complex128

N_TOP = 1.0
N_BOT = 1.0
THETA = 30 * np.pi / 180
PHI = 30 * np.pi / 180    # see README: RETICOLO's angle_delta0 is -30 deg

PERIOD = 1000e-9
THICKNESS = (500e-9, 500e-9)     # grating layer, then slab
GRATING_WIDTH = 500e-9
GRATING_CENTER = 500e-9

N_X, N_Y, N_Z = 1.5, 2.0, 2.5
N_AIR = [1.0, 1.0, 1.0]
N_GRATING = [N_X, N_Y, N_Z]
N_SLAB = [N_X, N_Y, N_Z]

FTO = [100]               # RETICOLO nn = 100

METHODS = ('discrete', 'continuous', 'vector')
POLS = {0: 'TE', 1: 'TM'}

# --------------------------------------------------------------------------- #
# sweep
# --------------------------------------------------------------------------- #

WAVELENGTHS_NM = np.arange(500, 701, 1, dtype=float)

# Subset for the default (non-slow) live test. Not arbitrary -- these are the
# wavelengths where the recorded run deviates most from RETICOLO, so a
# regression shows up here first:
#   540 nm  worst TM discrete deviation   (1.03e-05)
#   565 nm  worst TE discrete deviation   (6.25e-06)
#   639 nm  worst continuous/vector deviation (6.2e-11) and worst energy residual
#   700 nm  sweep endpoint
SMOKE_WAVELENGTHS_NM = (540.0, 565.0, 639.0, 700.0)

# --------------------------------------------------------------------------- #
# tolerances
# --------------------------------------------------------------------------- #
# MEASURED_* are what the committed data actually shows (recomputed from the
# files in this directory). TOL_* are what the tests assert -- each is the
# measured value with roughly a decade of headroom. Never widen a TOL_ without
# recording why here.

TOL_ENERGY = 1e-9
MEASURED_ENERGY = {          # max |R + T - 1| over the sweep
    'reticolo': 9.81e-11,
    'discrete': 9.76e-12,
    'continuous': 1.14e-11,
    'vector': 2.93e-11,
}

TOL_REFERENCE = {'discrete': 1e-4, 'continuous': 1e-8, 'vector': 1e-6}
MEASURED_REFERENCE = {       # max(|dR|, |dT|) vs RETICOLO, worst over both pols
    'discrete': 1.03e-5,     # staircase error of the discrete Fourier expansion
    'continuous': 6.20e-11,
    'vector': 6.24e-11,
}

TOL_METHOD_CROSS = 1e-9      # continuous vs vector: same convolution matrix, two ways
MEASURED_METHOD_CROSS = 1.05e-11

# Live meent vs the recorded run.
#
# Measured 2026-08-04 on the merged main (58ff99f) -- Linux, torch 2.13.0+cpu,
# numpy 2.4.6 -- against a recorded sweep produced on Windows with a different
# LAPACK build: worst deviation 4.94e-12 over the smoke subset. Cross-platform
# eigendecomposition agrees far better than the decade of headroom originally
# guessed here (1e-9), so this is tightened to 1e-10.
#
# It survived the unified iso/aniso solver path (dc4b077) and the evanescent kz
# branch fix (1c3017b) unchanged, which is the useful part: those refactors did
# not move this case.
TOL_RECORDED = 1e-10


# --------------------------------------------------------------------------- #
# meent construction
# --------------------------------------------------------------------------- #

def build_raster():
    """(layers, H, W, 3) -- the trailing 3 is what marks the ucell anisotropic.

    Columns are the 250 nm quarters of the period, matching RETICOLO's
    xb = 250:250:1000 with L1 = [air, bar, bar, air]: the bar spans 250-750 nm,
    i.e. width 500 nm centred at 500 nm.
    """
    layer1 = [N_AIR, N_GRATING, N_GRATING, N_AIR]
    layer2 = [N_SLAB] * 4
    return np.array([[layer1], [layer2]], dtype=float)


def build_vector():
    """The same structure as vector instructions instead of a raster."""
    return [
        [N_AIR, [['rectangle', GRATING_CENTER, GRATING_CENTER,
                  GRATING_WIDTH, PERIOD, N_GRATING]]],
        [N_SLAB, []],
    ]


def build(method, pol, wavelength_nm):
    """kwargs for meent.call_mee reproducing one point of the sweep."""
    if method not in METHODS:
        raise ValueError(f'unknown method {method!r}; expected one of {METHODS}')

    kwargs = dict(
        backend=BACKEND, pol=pol,
        n_top=N_TOP, n_bot=N_BOT, theta=THETA, phi=PHI,
        fto=list(FTO), wavelength=wavelength_nm * 1e-9,
        period=[PERIOD], thickness=list(THICKNESS),
        type_complex=TYPE_COMPLEX,
    )
    if method == 'vector':
        kwargs['ucell'] = build_vector()
    else:
        kwargs['ucell'] = build_raster()
        kwargs['fourier_type'] = 0 if method == 'discrete' else 1
    return kwargs


def observables(result):
    """(R, T) from a meent result -- total reflected and transmitted power.

    RETICOLO's res2 gives per-order efficiencies and the .m script sums them, so
    meent's orders are summed the same way. `pol` selects the incident
    polarization through psi, and `.res` for pol=0/1 corresponds to RETICOLO's
    TEinc/TMinc from its single conical solve.
    """
    res = result.res
    return float(res.de_ri.sum()), float(res.de_ti.sum())


# --------------------------------------------------------------------------- #
# data files
# --------------------------------------------------------------------------- #

def reference_file(pol):
    return f'RETICOLO_{NAME}_{POLS[pol]}.txt'


def recorded_file(pol, method):
    return f'Meent_{NAME}_{POLS[pol]}_{method}.txt'
