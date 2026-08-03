"""Does it run, is the shape right, is it the same twice.

Fast, tiny cases only (low fto). No physics claims -- those live in
test_physics.py. The point of this file is that when a physics test fails you
can tell instantly whether the solver merely crashed.
"""

import numpy as np
import pytest


class TestRuns:

    def test_conv_solve_runs(self, backend, fast_option):
        """Assert: returns a Result*, no exception, no NaN/inf anywhere."""
        pytest.skip('TODO: implement')

    @pytest.mark.parametrize('n_layers', [1, 2, 5])
    def test_multilayer_stack(self, backend, n_layers):
        """Assert: runs for a stack of n layers; layer_info_list has n entries.
        Why: solve_* iterates the stack in reverse; an off-by-one drops a layer.
        """
        pytest.skip('TODO: implement')

    def test_fto_zero(self, backend):
        """fto=0 -> only the zeroth order (ff=1, a 1x1 problem).
        Why: degenerate shapes are where reshape/diag calls break.
        """
        pytest.skip('TODO: implement')

    def test_zero_thickness_layer(self, backend):
        """A layer of thickness 0 must not produce NaN (exp(0) path)."""
        pytest.skip('TODO: implement')

    def test_absorbing_layer(self, backend):
        """Complex ucell entries run without warning."""
        pytest.skip('TODO: implement')


class TestShapes:

    def test_1d_result_shape(self, backend, option_1d_te):
        """Assert: de_ri.shape == (1, 2*fto[0]+1)."""
        pytest.skip('TODO: implement')

    def test_2d_result_shape(self, backend, option_2d):
        """Assert: de_ri.shape == (2*fto[1]+1, 2*fto[0]+1)."""
        pytest.skip('TODO: implement')

    def test_s_p_components_sum_to_total(self, backend, option):
        """Assert: de_ri == de_ri_s + de_ri_p (and same for de_ti).
        Why: the s/p split is a decomposition, not an independent computation.
        """
        pytest.skip('TODO: implement')

    def test_unused_polarization_component_is_zero(self, backend, option_1d_te):
        """1D TE: R_p, T_p, de_ri_p, de_ti_p are exactly zero (TM: mirrored).
        Why: transfer_1d_4 fills these with zeros by construction; a nonzero
        value means the wrong branch ran.
        """
        pytest.skip('TODO: implement')

    def test_efficiencies_are_real_and_nonnegative(self, backend, option):
        """de_* are real powers. Assert: real dtype (or zero imaginary part),
        all >= 0 (allowing a small negative epsilon from roundoff).
        """
        pytest.skip('TODO: implement')

    def test_amplitudes_are_complex(self, backend, option):
        """R_s/R_p/T_s/T_p are field amplitudes -- complex, not power."""
        pytest.skip('TODO: implement')


class TestSolverState:

    def test_layer_info_and_t1_populated(self, backend, option):
        """conv_solve fills layer_info_list and T1; calculate_field needs both.
        Assert: both are non-empty/non-None after the solve.
        """
        pytest.skip('TODO: implement')

    def test_state_is_reset_between_solves(self, backend, option):
        """solve_* starts with `self.layer_info_list = []`.
        Assert: two consecutive conv_solve calls leave the same number of layer
        entries (not double).
        Why: a leak here silently corrupts the field computation of the second run.
        """
        pytest.skip('TODO: implement')

    def test_repeated_solve_is_deterministic(self, backend, option):
        """Assert: bit-identical (or within 1e-14) results across repeated calls."""
        pytest.skip('TODO: implement')

    def test_mutating_an_option_changes_the_result(self, backend, option_1d_te):
        """The QA scripts mutate attributes between solves (mee.phi = 0, mee.ucell = ...).

        Assert: setting mee.wavelength / mee.ucell / mee.fourier_type and
        re-solving actually changes the result -- no stale cached conv matrices.
        Why: this reassignment pattern is the documented usage; it must be safe.
        """
        pytest.skip('TODO: implement')
