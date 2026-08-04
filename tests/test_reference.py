"""Validation against external solvers.

Every case in `tests/reference_cases/` is run through the same battery. Adding a
case adds tests automatically — see tests/reference_cases/README.md.

Two independent things are checked, and the distinction matters:

* **data-only tests** assert properties of the committed numbers. They need
  numpy and nothing else, so they run on any checkout, on any branch, with no
  optional backend installed. They are what preserves the experiment's finding
  after the person who ran it has moved on.
* **live tests** re-solve the structure with the meent in the working tree and
  compare against the reference. They need the backend and any feature the case
  declares, and skip with a specific reason when the checkout cannot do it.

A case whose feature has not been merged yet still contributes its data-only
tests. That is deliberate: the reference data can land before the code it
validates.
"""

import numpy as np
import pytest

from reference_cases import _loader

pytestmark = pytest.mark.reference

CASES = _loader.discover()
CASE_IDS = [c.NAME for c in CASES]


def _pol_method_params(cases):
    """(case, pol, method) for every case, as pytest params."""
    out = []
    for case in cases:
        for pol in case.POLS:
            for method in case.METHODS:
                out.append(pytest.param(case, pol, method,
                                        id=f'{case.NAME}-{case.POLS[pol]}-{method}'))
    return out


def _pol_params(cases):
    out = []
    for case in cases:
        for pol in case.POLS:
            out.append(pytest.param(case, pol, id=f'{case.NAME}-{case.POLS[pol]}'))
    return out


def _live_params(cases):
    """(case, pol, method, wavelength) over each case's smoke subset."""
    out = []
    for case in cases:
        for pol in case.POLS:
            for method in case.METHODS:
                for wl in case.SMOKE_WAVELENGTHS_NM:
                    out.append(pytest.param(
                        case, pol, method, wl,
                        id=f'{case.NAME}-{case.POLS[pol]}-{method}-{wl:.0f}nm'))
    return out


@pytest.fixture(scope='session')
def live(request):
    """Solve one point with the meent in the working tree, or skip with a reason.

    Session-scoped so the capability probe runs once, not once per wavelength.
    """
    def _live(case, method, pol, wavelength_nm):
        ok, reason = _loader.check_requirements(case, case.BACKEND)
        if not ok:
            pytest.skip(f'{case.NAME}: {reason}')
        import meent
        mee = meent.call_mee(**case.build(method, pol, wavelength_nm))
        return case.observables(mee.conv_solve())
    return _live


# --------------------------------------------------------------------------- #
# data-only: the committed numbers
# --------------------------------------------------------------------------- #

class TestReferenceData:
    """Properties of the external solver's own output.

    If these fail, the reference itself is suspect and every comparison against
    it is meaningless — so they run first and independently of meent.
    """

    @pytest.mark.parametrize('case, pol', _pol_params(CASES))
    def test_reference_conserves_energy(self, case, pol):
        """R + T == 1 for a lossless, non-leaking structure.

        This never mentions meent. It is an absolute check that the reference
        data is physical, and it is the reason a disagreement can be attributed
        rather than just observed.
        """
        table = _loader.reference_table(case, pol)
        lam, residual = _loader.energy_residual(table)
        assert len(residual), 'reference table has no finite rows'
        worst = np.argmax(np.abs(residual))
        assert abs(residual[worst]) <= case.TOL_ENERGY, (
            f'{case.NAME} {case.POLS[pol]}: |R+T-1| = {abs(residual[worst]):.3e} '
            f'at {lam[worst]:.0f} nm, tolerance {case.TOL_ENERGY:g}'
        )

    @pytest.mark.parametrize('case, pol', _pol_params(CASES))
    def test_reference_is_complete(self, case, pol):
        """No NaN, and the full declared sweep is present."""
        table = _loader.reference_table(case, pol)
        assert np.isfinite(table).all(), (
            f'{case.NAME} {case.POLS[pol]}: reference contains non-finite entries at '
            f'{table[~np.isfinite(table).all(axis=1)][:, 0]} nm'
        )
        np.testing.assert_allclose(
            table[:, _loader.COL_LAMBDA], case.WAVELENGTHS_NM, atol=1e-6,
            err_msg=f'{case.NAME} {case.POLS[pol]}: reference sweep does not match '
                    f'case.WAVELENGTHS_NM',
        )

    @pytest.mark.parametrize('case, pol', _pol_params(CASES))
    def test_efficiencies_are_physical(self, case, pol):
        """0 <= R, T <= 1. Cheap, and catches a mis-parsed column order."""
        table = _loader.reference_table(case, pol)
        for col, name in ((_loader.COL_R, 'R'), (_loader.COL_T, 'T')):
            values = table[:, col]
            assert values.min() >= -1e-12, f'{name} < 0 at {table[values.argmin(), 0]:.0f} nm'
            assert values.max() <= 1 + 1e-12, f'{name} > 1 at {table[values.argmax(), 0]:.0f} nm'


class TestRecordedData:
    """The meent run archived with the experiment.

    These assert the finding the experiment established, using only committed
    numbers. They keep working when the feature is unavailable, when torch is
    not installed, and on a branch where the solver cannot run at all.
    """

    @pytest.mark.parametrize('case, pol, method', _pol_method_params(CASES))
    def test_recorded_conserves_energy(self, case, pol, method):
        table = _loader.recorded_table(case, pol, method)
        lam, residual = _loader.energy_residual(table)
        worst = np.argmax(np.abs(residual))
        assert abs(residual[worst]) <= case.TOL_ENERGY, (
            f'{case.NAME} {case.POLS[pol]} {method}: |R+T-1| = '
            f'{abs(residual[worst]):.3e} at {lam[worst]:.0f} nm'
        )

    @pytest.mark.parametrize('case, pol, method', _pol_method_params(CASES))
    def test_recorded_matches_reference(self, case, pol, method):
        """**The archived scientific claim**: meent agreed with the external
        solver, to this tolerance, at every wavelength.

        This is the single most valuable assertion in the file. It does not
        depend on the current checkout being able to run the case at all, so the
        result survives refactors, branch churn and missing dependencies.
        """
        ref, rec = _loader.align(_loader.reference_table(case, pol),
                                 _loader.recorded_table(case, pol, method))
        assert len(ref), f'{case.NAME}: no overlapping wavelengths'

        d_r = np.abs(ref[:, _loader.COL_R] - rec[:, _loader.COL_R])
        d_t = np.abs(ref[:, _loader.COL_T] - rec[:, _loader.COL_T])
        deviation = np.maximum(d_r, d_t)
        tol = case.TOL_REFERENCE[method]
        worst = int(np.argmax(deviation))

        assert deviation[worst] <= tol, (
            f'{case.NAME} {case.POLS[pol]} {method}: worst deviation '
            f'{deviation[worst]:.3e} at {ref[worst, 0]:.0f} nm exceeds {tol:g} '
            f'({int((deviation > tol).sum())} of {len(deviation)} points over tolerance)'
        )

    @pytest.mark.parametrize('case, pol, method', _pol_method_params(CASES))
    def test_measured_deviation_is_still_accurate(self, case, pol, method):
        """`case.MEASURED_REFERENCE` documents what the data actually shows.

        Assert it has not drifted from the files. A mismatch means someone
        replaced the data without updating the case notes — the numbers in the
        README would then be describing a run that no longer exists.
        """
        ref, rec = _loader.align(_loader.reference_table(case, pol),
                                 _loader.recorded_table(case, pol, method))
        deviation = np.maximum(
            np.abs(ref[:, _loader.COL_R] - rec[:, _loader.COL_R]),
            np.abs(ref[:, _loader.COL_T] - rec[:, _loader.COL_T]),
        ).max()
        documented = case.MEASURED_REFERENCE[method]
        assert deviation <= documented * 1.5, (
            f'{case.NAME} {case.POLS[pol]} {method}: measured {deviation:.3e} but '
            f'case.MEASURED_REFERENCE says {documented:.3e} — update the case notes'
        )

    @pytest.mark.parametrize('case', CASES, ids=CASE_IDS)
    def test_continuous_and_vector_agree(self, case):
        """Two routes to the same convolution matrix must land in the same place.

        Independent of any external reference: it compares meent to meent, so it
        isolates the modeling layer from the physics.
        """
        if not {'continuous', 'vector'} <= set(case.METHODS):
            pytest.skip('case does not provide both continuous and vector runs')
        for pol in case.POLS:
            a, b = _loader.align(_loader.recorded_table(case, pol, 'continuous'),
                                 _loader.recorded_table(case, pol, 'vector'))
            deviation = np.maximum(
                np.abs(a[:, _loader.COL_R] - b[:, _loader.COL_R]),
                np.abs(a[:, _loader.COL_T] - b[:, _loader.COL_T]),
            ).max()
            assert deviation <= case.TOL_METHOD_CROSS, (
                f'{case.NAME} {case.POLS[pol]}: continuous vs vector {deviation:.3e} '
                f'exceeds {case.TOL_METHOD_CROSS:g}'
            )


# --------------------------------------------------------------------------- #
# live: this checkout, right now
# --------------------------------------------------------------------------- #

class TestLive:
    """Re-solve and compare. Skips with a reason when the checkout cannot."""

    @pytest.mark.parametrize('case, pol, method, wavelength_nm', _live_params(CASES))
    def test_live_matches_reference(self, case, pol, method, wavelength_nm, live):
        """meent as it stands today vs the external solver, at the wavelengths
        where the recorded run deviated most.
        """
        r, t = live(case, method, pol, wavelength_nm)
        ref = _loader.reference_table(case, pol)
        row = ref[np.isclose(ref[:, _loader.COL_LAMBDA], wavelength_nm, atol=1e-6)]
        assert len(row) == 1, f'{wavelength_nm} nm not in the reference sweep'
        tol = case.TOL_REFERENCE[method]

        np.testing.assert_allclose(
            [r, t], row[0, [_loader.COL_R, _loader.COL_T]], rtol=0, atol=tol,
            err_msg=f'{case.NAME} {case.POLS[pol]} {method} at {wavelength_nm:.0f} nm',
        )

    @pytest.mark.parametrize('case, pol, method, wavelength_nm', _live_params(CASES))
    def test_live_matches_recorded(self, case, pol, method, wavelength_nm, live):
        """meent today vs meent at the time of the experiment.

        Tighter than the reference comparison, and a different question: not "is
        meent right" but "has meent changed". A failure here with
        test_live_matches_reference still passing means a real but sub-tolerance
        drift — worth understanding before it grows.
        """
        r, t = live(case, method, pol, wavelength_nm)
        rec = _loader.recorded_table(case, pol, method)
        row = rec[np.isclose(rec[:, _loader.COL_LAMBDA], wavelength_nm, atol=1e-6)]
        assert len(row) == 1, f'{wavelength_nm} nm not in the recorded sweep'

        np.testing.assert_allclose(
            [r, t], row[0, [_loader.COL_R, _loader.COL_T]],
            rtol=0, atol=case.TOL_RECORDED,
            err_msg=f'{case.NAME} {case.POLS[pol]} {method} at {wavelength_nm:.0f} nm '
                    f'drifted from the recorded run',
        )

    @pytest.mark.parametrize('case, pol, method, wavelength_nm', _live_params(CASES))
    def test_live_conserves_energy(self, case, pol, method, wavelength_nm, live):
        """The invariant, on a fresh solve. Needs no stored data at all."""
        r, t = live(case, method, pol, wavelength_nm)
        assert abs(r + t - 1) <= case.TOL_ENERGY

    @pytest.mark.slow
    @pytest.mark.parametrize('case, pol, method', _pol_method_params(CASES))
    def test_full_sweep_matches_reference(self, case, pol, method, live):
        """The whole experiment, re-run. ~10 minutes per case at fto=100.

        This is what to run before releasing, or after touching the solver. The
        smoke subset above is what runs day to day.
        """
        ref = _loader.reference_table(case, pol)
        tol = case.TOL_REFERENCE[method]
        failures = []
        for row in ref:
            wavelength_nm = row[_loader.COL_LAMBDA]
            r, t = live(case, method, pol, wavelength_nm)
            deviation = max(abs(r - row[_loader.COL_R]), abs(t - row[_loader.COL_T]))
            if deviation > tol:
                failures.append((wavelength_nm, deviation))
        assert not failures, (
            f'{case.NAME} {case.POLS[pol]} {method}: {len(failures)} of {len(ref)} '
            f'points over {tol:g}; worst {max(f[1] for f in failures):.3e} at '
            f'{max(failures, key=lambda f: f[1])[0]:.0f} nm'
        )


# --------------------------------------------------------------------------- #
# the registry itself
# --------------------------------------------------------------------------- #

class TestCaseDefinitions:

    def test_at_least_one_case_is_registered(self):
        """A silent discovery failure would turn this whole file into a no-op."""
        assert CASES, 'no reference cases discovered in tests/reference_cases/'

    @pytest.mark.parametrize('case', CASES, ids=CASE_IDS)
    def test_case_declares_the_required_interface(self, case):
        for attr in ('NAME', 'DESCRIPTION', 'REQUIRES', 'BACKEND', 'METHODS', 'POLS',
                     'WAVELENGTHS_NM', 'SMOKE_WAVELENGTHS_NM', 'TOL_ENERGY',
                     'TOL_REFERENCE', 'MEASURED_REFERENCE', 'TOL_RECORDED',
                     'build', 'observables', 'reference_file', 'recorded_file'):
            assert hasattr(case, attr), f'{case.NAME} is missing {attr}'

    @pytest.mark.parametrize('case', CASES, ids=CASE_IDS)
    def test_declared_files_exist(self, case):
        for pol in case.POLS:
            assert (case.DIR / 'reference' / case.reference_file(pol)).exists()
            for method in case.METHODS:
                assert (case.DIR / 'recorded' / case.recorded_file(pol, method)).exists()

    @pytest.mark.parametrize('case', CASES, ids=CASE_IDS)
    def test_every_method_has_a_tolerance(self, case):
        for method in case.METHODS:
            assert method in case.TOL_REFERENCE, f'{case.NAME}: no tolerance for {method}'
            assert method in case.MEASURED_REFERENCE, f'{case.NAME}: no measurement for {method}'

    @pytest.mark.parametrize('case', CASES, ids=CASE_IDS)
    def test_tolerances_have_headroom_over_measurements(self, case):
        """A tolerance at or below what the data already shows will fail the
        first time it is run. Catches a copy-paste when adding a case.
        """
        for method in case.METHODS:
            assert case.TOL_REFERENCE[method] > case.MEASURED_REFERENCE[method], (
                f'{case.NAME}: TOL_REFERENCE[{method!r}] is not above the measured '
                f'deviation'
            )

    @pytest.mark.parametrize('case', CASES, ids=CASE_IDS)
    def test_smoke_wavelengths_are_in_the_sweep(self, case):
        for wavelength_nm in case.SMOKE_WAVELENGTHS_NM:
            assert np.any(np.isclose(case.WAVELENGTHS_NM, wavelength_nm, atol=1e-6)), (
                f'{case.NAME}: smoke wavelength {wavelength_nm} is not in WAVELENGTHS_NM'
            )

    @pytest.mark.parametrize('case', CASES, ids=CASE_IDS)
    def test_build_returns_call_mee_kwargs(self, case):
        """Runs without meent installed — just checks the shape of the dict."""
        for method in case.METHODS:
            kwargs = case.build(method, 0, float(case.SMOKE_WAVELENGTHS_NM[0]))
            assert kwargs['backend'] == case.BACKEND
            assert 'ucell' in kwargs and kwargs['ucell'] is not None
            assert kwargs['wavelength'] > 0
