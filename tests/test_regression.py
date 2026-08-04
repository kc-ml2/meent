"""Golden-value regression and external references.

The tests above check internal consistency and physical invariants. This file
checks that today's numbers equal yesterday's, and -- where a reference exists --
that they equal an independent solver's.

Rule for this file: a failure is a *finding*, not a chore. Do not regenerate a
golden file to make a test pass without understanding what moved.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.regression


class TestLiteratureCases:
    """examples/rcwa/ reproduces published structures. Those are the natural
    regression cases: physically meaningful, already trusted, already written."""

    @pytest.mark.parametrize('case', [
        'moharam_1D_TE',
        'moharam_1D_TM',
        'moharam_1D_conical',
        'moharam_2D',
        'lalanne_1D_TM',
        'lalanne_1D_conical',
    ])
    def test_matches_golden(self, backend, case, golden):
        """Setup: the option dict from the matching examples/rcwa/ script.

        Assert: de_ri/de_ti match tests/data/<case>.npz to ~1e-10 (complex128).
        Why: catches any change in physics, however small, across a refactor.
        TODO: factor the option dicts out of examples/rcwa/*.py so the examples
        and the tests cannot drift apart. Importing them directly would be
        better than copying -- the scripts currently run on import, so they need
        a `if __name__ == '__main__'` guard first.
        """
        pytest.skip('TODO: implement')

    @pytest.mark.parametrize('case', ['moharam_1D_TE', 'moharam_2D'])
    def test_matches_published_values(self, case):
        """Where the paper prints efficiencies, assert against the *paper*, not
        against our own stored output.

        Assert: agreement to the precision quoted in the source.
        Why: a golden file only proves we did not change. This proves we were
        right in the first place -- worth the effort for at least one case per
        formulation. Cite the paper, table and row in the test.
        """
        pytest.skip('TODO: transcribe published values, then implement')


# Validation against external solvers (RETICOLO, GRCWA, TORCWA) is NOT here.
# It has its own structure -- reference output, meent output, provenance and a
# rebuild recipe per experiment -- under tests/reference_cases/, driven by
# tests/test_reference.py. Add a case there rather than a stub here; see
# tests/reference_cases/README.md.
#
# benchmarks/reti_meent_{1D,1Dc,2D}.py and benchmarks/interface/ are the natural
# sources for the next few cases.


class TestGoldenInfrastructure:

    def test_golden_files_exist_for_every_case(self):
        """Assert: every parametrized case above has a file in tests/data/.
        Why: a missing golden file otherwise turns into a skip, and a skipped
        regression test protects nothing.
        """
        pytest.skip('TODO: implement')

    def test_regen_flag_fails_the_run(self):
        """--regen-golden must rewrite the files AND fail, so a regeneration run
        can never be mistaken for a passing one in CI.
        """
        pytest.skip('TODO: implement (see conftest.golden)')

    def test_golden_files_record_provenance(self):
        """Each .npz should carry the meent version, backend, dtype and the option
        dict that produced it.
        Assert: those keys are present.
        Why: a bare array of numbers with no provenance is not a reference.
        """
        pytest.skip('TODO: implement')
