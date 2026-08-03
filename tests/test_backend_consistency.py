"""numpy vs jax vs torch.

Three independent implementations of the same algorithm, so any disagreement is
a bug in at least one of them. This is the direct successor to
QA/rcwa_backend_consistency.py -- same option dicts (lifted into conftest.OPTIONS),
same quantities, but asserted instead of printed.

Cost note: every test here runs the full solve three times. Keep the option set
small and lean on the cheaper layers (test_fourier.py's conv-matrix comparison)
to localize failures.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.backend_consistency

# QA/rcwa_backend_consistency.py compares this exact attribute list.
CHECK_ATTRS = [
    'R_s', 'R_p', 'T_s', 'T_p',
    'de_ri_s', 'de_ri_p', 'de_ri',
    'de_ti_s', 'de_ti_p', 'de_ti',
]


@pytest.fixture(params=[(0, 1), (0, 2), (1, 2)], ids=['numpy-jax', 'numpy-torch', 'jax-torch'])
def backend_pair(request):
    """Pairs of installed backends. TODO: skip the pair if either dep is absent."""
    pytest.skip('TODO: implement (see conftest._installed)')


class TestConvSolve:

    @pytest.mark.parametrize('attr', CHECK_ATTRS)
    def test_res_channel_matches(self, backend_pair, option, attr):
        """Assert: ||a - b|| / a.size below tolerance, for result.res.

        Why: R/T amplitudes are compared alongside efficiencies deliberately --
        amplitudes carry phase, so they catch a sign or conjugation difference
        that squaring into an efficiency would hide.
        """
        pytest.skip('TODO: implement')

    @pytest.mark.parametrize('attr', CHECK_ATTRS)
    def test_te_inc_channel_matches(self, backend_pair, option_2d, attr):
        """Same for result.res_te_inc (2D / conical cases only)."""
        pytest.skip('TODO: implement')

    @pytest.mark.parametrize('attr', CHECK_ATTRS)
    def test_tm_inc_channel_matches(self, backend_pair, option_2d, attr):
        """Same for result.res_tm_inc."""
        pytest.skip('TODO: implement')

    def test_vector_modeling_matches(self, backend_pair, option_2d):
        """The vector modeler is reimplemented per backend (geometry code, not
        just array ops), so it needs its own cross-check.
        """
        pytest.skip('TODO: implement')


class TestField:

    def test_field_cell_matches(self, backend_pair, fast_option):
        """Assert: calculate_field(res_x=50, res_z=50) agrees across backends.
        Why: the field reconstruction uses the eigenvectors directly, unlike the
        efficiencies -- so it is the only test here sensitive to eigen-ordering
        differences between LAPACK builds. Expect this to be the loosest
        tolerance in the file; if it needs to be very loose, say why here.
        """
        pytest.skip('TODO: implement')


class TestNumerics:

    def test_tolerance_is_documented(self):
        """Placeholder for the decision this file rests on.

        Record in conftest.TOL what agreement is actually achievable per pair and
        per dtype, and why. Do not tune the number until tests pass -- measure it
        once on a known-good commit and treat later drift as a finding.
        """
        pytest.skip('TODO: measure and document')

    @pytest.mark.slow
    def test_ill_conditioned_case_still_agrees(self, backend_pair):
        """Setup: near-degenerate eigenvalues (e.g. a nearly uniform layer, where
        `perturbation` matters).
        Assert: backends still agree, or document the divergence explicitly.
        Why: this is where they will actually differ, and knowing the size of
        that difference is more useful than a passing test on easy cases.
        """
        pytest.skip('TODO: implement')
