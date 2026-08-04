"""Discovery and loading for reference cases.

A *reference case* is one validation experiment: a structure solved by an
established external code (RETICOLO, GRCWA, ...), the numbers that code
produced, and enough information to make meent solve the same thing.

Adding a case never requires touching test code -- drop a directory next to this
file with a `case.py` in it and `tests/test_reference.py` picks it up. See
README.md in this directory for the recipe.

Nothing here imports meent, torch or jax at module level: the data-only tests
must run on a bare numpy install.
"""

import importlib.util
from pathlib import Path

import numpy as np

CASES_DIR = Path(__file__).parent

# Columns of every result table, reference and recorded alike.
COL_LAMBDA, COL_R, COL_T = 0, 1, 2


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #

def discover():
    """Every case directory, as imported modules, sorted by name."""
    cases = []
    for directory in sorted(CASES_DIR.iterdir()):
        if directory.is_dir() and (directory / 'case.py').exists():
            cases.append(load_case(directory))
    return cases


def load_case(directory):
    """Import <directory>/case.py and attach its location as `case.DIR`."""
    directory = Path(directory)
    spec = importlib.util.spec_from_file_location(
        f'meent_reference_case_{directory.name}', directory / 'case.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DIR = directory
    return module


# --------------------------------------------------------------------------- #
# data files
# --------------------------------------------------------------------------- #

def load_table(path):
    """Read a `lambda_nm, R, T` csv into an (N, 3) array.

    Kept deliberately strict: a malformed file should raise here rather than
    produce NaNs that a test then skips over.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    table = np.genfromtxt(path, delimiter=',', skip_header=1)
    if table.ndim == 1:
        table = table.reshape(1, -1)
    if table.shape[1] != 3:
        raise ValueError(f'{path}: expected 3 columns (lambda_nm, R, T), got {table.shape[1]}')
    return table


def reference_table(case, pol):
    return load_table(case.DIR / 'reference' / case.reference_file(pol))


def recorded_table(case, pol, method):
    return load_table(case.DIR / 'recorded' / case.recorded_file(pol, method))


def align(table_a, table_b, atol=1e-6):
    """Match two tables on their wavelength column.

    Returns (a_rows, b_rows) restricted to wavelengths present in both and
    finite in both. Reference sweeps and meent sweeps are generated separately,
    so never assume the rows line up positionally.
    """
    lam_a, lam_b = table_a[:, COL_LAMBDA], table_b[:, COL_LAMBDA]
    idx_a, idx_b = [], []
    for i, lam in enumerate(lam_a):
        j = np.where(np.isclose(lam_b, lam, atol=atol, rtol=0))[0]
        if j.size:
            idx_a.append(i)
            idx_b.append(j[0])
    a, b = table_a[idx_a], table_b[idx_b]
    finite = np.all(np.isfinite(a), axis=1) & np.all(np.isfinite(b), axis=1)
    return a[finite], b[finite]


def energy_residual(table):
    """R + T - 1 for every finite row. Zero for a lossless, non-leaking stack."""
    finite = np.all(np.isfinite(table), axis=1)
    rows = table[finite]
    return rows[:, COL_LAMBDA], rows[:, COL_R] + rows[:, COL_T] - 1.0


# --------------------------------------------------------------------------- #
# capability probes
# --------------------------------------------------------------------------- #
# A case declares REQUIRES = {'backends': [...], 'features': [...]}. Live tests
# skip -- with a specific reason -- when the running checkout cannot do it.
#
# This is what lets reference data be committed *before* the feature it
# validates is merged: the data-only tests still assert the archived result,
# and the live tests announce exactly what is missing.

def backend_available(backend):
    module = {0: 'numpy', 1: 'jax', 2: 'torch'}[backend]
    if importlib.util.find_spec(module) is None:
        return False, f'{module} is not installed'
    return True, ''


def _probe_anisotropy_diagonal(backend):
    """Can this checkout solve a diagonally anisotropic (nx, ny, nz) ucell?

    Support landed on the `feature/anisotropic-diagonal-torch` branch (torch
    only). On a checkout without it the 4-D ucell is rejected by the ucell
    setter, so a one-wavelength solve is the honest test.
    """
    try:
        import meent
        ucell = np.array([[[[1.0, 1.0, 1.0], [1.5, 2.0, 2.5]]]])
        mee = meent.call_mee(
            backend=backend, pol=0, n_top=1, n_bot=1, theta=0.1, phi=0.1,
            fto=[1], wavelength=1.0, period=[2.0], thickness=[1.0], ucell=ucell,
        )
        mee.conv_solve()
    except Exception as exc:  # noqa: BLE001 -- any failure means "unsupported"
        return False, f'anisotropic ucell not supported by this checkout ({type(exc).__name__}: {exc})'
    return True, ''


FEATURE_PROBES = {
    'anisotropy_diagonal': _probe_anisotropy_diagonal,
}

_probe_cache = {}


def check_requirements(case, backend):
    """(ok, reason) for running `case` live on `backend`.

    Probe results are cached -- a probe runs a real solve, and every live test
    would otherwise pay for it.
    """
    requires = getattr(case, 'REQUIRES', {})

    backends = requires.get('backends')
    if backends and backend not in backends:
        return False, f'case declares backends={backends}, not {backend}'

    ok, reason = backend_available(backend)
    if not ok:
        return False, reason

    for feature in requires.get('features', []):
        key = (feature, backend)
        if key not in _probe_cache:
            probe = FEATURE_PROBES.get(feature)
            if probe is None:
                _probe_cache[key] = (False, f'unknown feature {feature!r} -- add a probe to _loader.FEATURE_PROBES')
            else:
                _probe_cache[key] = probe(backend)
        ok, reason = _probe_cache[key]
        if not ok:
            return False, reason

    return True, ''
