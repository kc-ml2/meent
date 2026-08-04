"""Shared fixtures for the meent test suite.

This file is deliberately the only place that knows about backend plumbing
(missing optional deps, tensor -> ndarray conversion, tolerances). Test modules
should read as physics, not as `if backend == 2: x = x.detach()`.
"""

import importlib.util

import numpy as np
import pytest

# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #

BACKEND_NAMES = {0: 'numpy', 1: 'jax', 2: 'torch'}
BACKEND_MODULES = {0: 'numpy', 1: 'jax', 2: 'torch'}


def _installed(module_name):
    return importlib.util.find_spec(module_name) is not None


def _requires(backend):
    """Skip marker for a backend whose optional dependency is absent."""
    return pytest.mark.skipif(
        not _installed(BACKEND_MODULES[backend]),
        reason=f'{BACKEND_MODULES[backend]} not installed',
    )


@pytest.fixture(
    params=[
        pytest.param(0, id='numpy'),
        pytest.param(1, id='jax', marks=[pytest.mark.jax, _requires(1)]),
        pytest.param(2, id='torch', marks=[pytest.mark.torch, _requires(2)]),
    ]
)
def backend(request):
    """Every installed backend, one at a time."""
    return request.param


@pytest.fixture(
    params=[
        pytest.param(1, id='jax', marks=[pytest.mark.jax, _requires(1)]),
        pytest.param(2, id='torch', marks=[pytest.mark.torch, _requires(2)]),
    ]
)
def ad_backend(request):
    """Differentiable backends only (numpy has no `grad`/`fit`)."""
    return request.param


def to_numpy(x):
    """Normalize a backend array to numpy.

    TODO: torch tensors need `.detach().cpu().numpy()`; jax arrays go through
    `np.asarray`. Nested containers (the `ResultSub*` objects) are handled by
    `result_as_dict` below.
    """
    raise NotImplementedError('TODO: implement backend -> ndarray conversion')


RESULT_ATTRS = (
    'R_s', 'R_p', 'T_s', 'T_p',
    'de_ri', 'de_ri_s', 'de_ri_p',
    'de_ti', 'de_ti_s', 'de_ti_p',
)


def result_as_dict(res_sub):
    """`ResultSub{Numpy,Jax,Torch}` -> {attr: ndarray}, for backend-agnostic asserts."""
    raise NotImplementedError('TODO: map RESULT_ATTRS through to_numpy')


# --------------------------------------------------------------------------- #
# tolerances
# --------------------------------------------------------------------------- #

# Placeholders. These are physics calls, not style calls -- see tests/README.md.
TOL = {
    'complex128': dict(rtol=1e-10, atol=1e-12),
    'complex64': dict(rtol=1e-4, atol=1e-6),
}


@pytest.fixture
def tol():
    """tol(type_complex) -> dict(rtol=..., atol=...) suitable for np.testing."""
    def _tol(type_complex=0):
        return TOL['complex128' if type_complex in (0, np.complex128) else 'complex64']
    return _tol


# --------------------------------------------------------------------------- #
# canonical option dicts
# --------------------------------------------------------------------------- #
# Lifted from QA/rcwa_backend_consistency.py so the QA scripts and the test
# suite exercise literally the same cases. Keys map 1:1 onto call_mee kwargs.
#
# Add new cases HERE, not inline in a test -- test_backend_consistency.py,
# test_physics.py and test_regression.py all parametrize over this registry.

def _ucell_1d_binary():
    return np.array([[[3, 3, 3, 3, 3, 1, 1, 1, 1, 1]]], dtype=float)


def _ucell_1d_multi():
    return np.array([[[3, 3, 3.3, 3, 3, 4, 1, 1, 1, 1.2, 1.1, 3, 2, 1.1]]], dtype=float)


def _ucell_vector_2d():
    """Vector (list-of-instructions) modeling; see ModelingNumpy.modeling_vector_instruction."""
    return [
        [1, [
            ['rectangle', 0 + 240, 120 + 240, 160, 80, 4, 1, 20, 20],
            ['rectangle', 0 + 240, -120 + 240, 160, 80, 4, 0, 0, 0],
            ['rectangle', 120 + 240, 0 + 240, 80, 160, 4, 0, 0, 0],
            ['rectangle', -120 + 240, 0 + 240, 80, 160, 4, 0, 0, 0],
        ]],
    ]


OPTIONS = {
    # id -> (option dict, expected _grating_type_assigned)
    '1d_te': (dict(
        pol=0, n_top=2, n_bot=1, theta=12 * np.pi / 180, phi=None, fto=0,
        period=[770], wavelength=777, thickness=[100], fourier_type=0,
        ucell=_ucell_1d_binary(),
    ), 0),

    '1d_tm_cfs': (dict(
        pol=1, n_top=1, n_bot=1.3, theta=0., phi=None, fto=40,
        period=[2000], wavelength=400, thickness=[1000], fourier_type=1,
        ucell=_ucell_1d_multi(),
    ), 0),

    '1d_conical_psi': (dict(
        psi=40 / 180 * np.pi, n_top=1, n_bot=1, theta=0., phi=12 * np.pi / 180,
        fto=80, period=[200], wavelength=1000, thickness=[100],
        fourier_type=0, enhanced_dfs=False, ucell=_ucell_1d_multi(),
    ), 1),

    '2d_raster': (dict(
        psi=10 / 180 * np.pi, n_top=1, n_bot=1.5, theta=30 * np.pi / 180, phi=0.,
        fto=[10, 10], period=[200, 600], wavelength=1000,
        thickness=[100, 111, 222, 102, 44], fourier_type=0, enhanced_dfs=True,
        ucell=None,  # TODO: seeded RNG -- np.random.default_rng(0).random((5, 20, 20)) * 3 + 1
    ), 2),

    '2d_vector': (dict(
        pol=0, n_top=2, n_bot=1, theta=12 * np.pi / 180, phi=0., fto=[5, 5],
        period=[770], wavelength=777, thickness=[100], fourier_type=0,
        ucell=_ucell_vector_2d(),
    ), 2),
}

# Small/fast subset for smoke tests and anything parametrized over 3 backends.
FAST_OPTION_IDS = ['1d_te', '2d_vector']


@pytest.fixture(params=sorted(OPTIONS), ids=sorted(OPTIONS))
def option(request):
    """Every canonical case as a fresh dict (never share mutable ucells)."""
    opt, _ = OPTIONS[request.param]
    return dict(opt)


@pytest.fixture(params=FAST_OPTION_IDS, ids=FAST_OPTION_IDS)
def fast_option(request):
    opt, _ = OPTIONS[request.param]
    return dict(opt)


@pytest.fixture
def option_1d_te():
    return dict(OPTIONS['1d_te'][0])


@pytest.fixture
def option_2d():
    return dict(OPTIONS['2d_vector'][0])


@pytest.fixture
def mee():
    """mee(backend, **option) -> configured Mee instance.

    Thin wrapper over meent.call_mee so tests don't repeat the import and so a
    future change to the entry point lands in one place.
    """
    def _mee(backend=0, **option):
        import meent
        return meent.call_mee(backend=backend, **option)
    return _mee


@pytest.fixture
def solve(mee):
    """solve(backend, **option) -> (mee, result). Convenience for one-shot cases."""
    def _solve(backend=0, **option):
        instance = mee(backend, **option)
        return instance, instance.conv_solve()
    return _solve


# --------------------------------------------------------------------------- #
# golden-value regression support
# --------------------------------------------------------------------------- #

def pytest_addoption(parser):
    parser.addoption(
        '--regen-golden',
        action='store_true',
        default=False,
        help='Rewrite tests/data golden files instead of comparing against them.',
    )


@pytest.fixture
def golden(request):
    """golden(name, values) -> compares against tests/data/<name>.npz, or rewrites it.

    TODO: load tests/data/<name>.npz and np.testing.assert_allclose; when
    --regen-golden is set, np.savez_compressed instead and fail the test so a
    regeneration run can never be mistaken for a passing run.

    Skips rather than raising: an unimplemented fixture that raises turns every
    test requesting it into an ERROR during setup, which reads as a broken suite
    rather than an unwritten one.
    """
    pytest.skip('TODO: implement golden load/compare/regen')
