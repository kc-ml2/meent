"""_BaseRCWA property setters and grating-type dispatch.

No simulation runs here -- these are pure input-normalization tests and should
stay in the millisecond range. Most user-facing "wrong answer" bugs in an RCWA
package are actually input-coercion bugs, so this layer is worth over-covering.
"""

import numpy as np
import pytest


class TestPolarization:
    """pol and psi are two views of one quantity and set each other."""

    @pytest.mark.parametrize('pol, expected_psi', [
        (0, np.pi / 2),   # full TE
        (1, 0.0),         # full TM
        (0.5, np.pi / 4),
    ])
    def test_pol_sets_psi(self, backend, pol, expected_psi):
        """psi = pi/2 * (1 - pol). Assert: mee.psi == expected."""
        pytest.skip('TODO: implement')

    def test_psi_sets_pol(self, backend):
        """pol = -(2*psi/pi - 1). Assert: round trip pol -> psi -> pol is identity."""
        pytest.skip('TODO: implement')

    @pytest.mark.parametrize('bad', [-0.1, 1.1, 2])
    def test_pol_out_of_range_raises(self, backend, bad):
        """Assert: ValueError. Why: pol is a ratio, not an angle."""
        pytest.skip('TODO: implement')

    def test_psi_none_leaves_pol_untouched(self, backend):
        """The psi setter early-returns on None, leaving whatever pol set.

        Assert: constructing with pol=1, psi=None leaves pol == 1 and psi == 0.
        Why: call_mee passes psi=None by default; this is the common path.
        """
        pytest.skip('TODO: implement')


class TestAngles:

    def test_theta_zero_is_perturbed(self, backend):
        """theta == 0 is replaced by `perturbation` (default 1e-20).

        Assert: mee.theta != 0 and abs(mee.theta) <= perturbation.
        Why: cos/kz normalization divides by quantities that vanish at exactly
        normal incidence; the perturbation is load-bearing, not cosmetic.
        """
        pytest.skip('TODO: implement')

    def test_theta_nonzero_passes_through(self, backend):
        pytest.skip('TODO: implement')

    def test_phi_none_stays_none(self, backend):
        """phi=None is the sentinel that selects the 1D TE/TM formulation.

        Assert: mee.phi is None (NOT array(None) or 0).
        Why: `phi=0` and `phi=None` are physically identical but pick different
        code paths -- see TestGratingTypeAssignment.
        """
        pytest.skip('TODO: implement')


class TestFto:

    @pytest.mark.parametrize('given, expected', [
        (5, [5, 0]),
        (5.9, [5, 0]),          # truncated by int()
        ([5], [5, 0]),
        ((5, 3), [5, 3]),
        (np.array([5, 3]), [5, 3]),
        (np.array(5), [5, 0]),
    ])
    def test_normalization(self, backend, given, expected):
        """fto is always stored as a 2-list of ints."""
        pytest.skip('TODO: implement')

    @pytest.mark.parametrize('bad', [[1, 2, 3], 'a', None])
    def test_invalid_raises(self, backend, bad):
        pytest.skip('TODO: implement')

    def test_ff_relation(self, backend):
        """Number of retained orders ff = 2*fto + 1 per axis.

        Assert: de_ri.shape == (2*fto[1]+1, 2*fto[0]+1) for a 2D case.
        Why: ties the stored fto to the observable output shape.
        """
        pytest.skip('TODO: implement')


class TestPeriodThickness:

    @pytest.mark.parametrize('given, expected', [
        (700, [700, 700]),
        ([700], [700, 700]),
        ([700, 300], [700, 300]),
        (np.array([700, 300]), [700, 300]),
    ])
    def test_period_broadcast(self, backend, given, expected):
        """A scalar or length-1 period means a square/1D-periodic cell."""
        pytest.skip('TODO: implement')

    @pytest.mark.parametrize('given, expected_len', [(100, 1), ([100, 200], 2)])
    def test_thickness_normalization(self, backend, given, expected_len):
        pytest.skip('TODO: implement')

    def test_thickness_length_must_match_ucell_layers(self, backend):
        """len(thickness) must equal ucell.shape[0].

        Assert: current behavior on mismatch (solve_* iterates over thickness and
        indexes epx_conv_all by the same index). Decide and pin: raise at set
        time, or IndexError at solve time.
        Why: a silent truncation here would drop a layer from the stack.
        """
        pytest.skip('TODO: decide intended behavior, then implement')

    @pytest.mark.parametrize('bad', ['x', None, {}])
    def test_invalid_types_raise(self, backend, bad):
        pytest.skip('TODO: implement')


class TestDtype:

    @pytest.mark.parametrize('given', [0, 1])
    def test_type_complex_selects_matching_float_and_int(self, backend, given):
        """complex128 -> float64/int64; complex64 -> float32/int32."""
        pytest.skip('TODO: implement')

    def test_invalid_type_complex_raises(self, backend):
        pytest.skip('TODO: implement')

    def test_changing_type_complex_recasts_existing_attrs(self, backend):
        """The setter re-assigns theta, phi, psi, fto, thickness after switching.

        Assert: build with complex128, set type_complex = 1, then every stored
        array reports the 32-bit dtype.
        Why: a stale float64 thickness inside a complex64 run is the exact class
        of bug the README's 8-digit warning is about.
        """
        pytest.skip('TODO: implement')


class TestUcellAndGratingType:

    def test_ndarray_ucell_is_raster(self, backend, option_1d_te):
        """Assert: modeling_type_assigned == 0."""
        pytest.skip('TODO: implement')

    def test_list_ucell_is_vector(self, backend, option_2d):
        """Assert: modeling_type_assigned == 1, and ucell stored unmodified."""
        pytest.skip('TODO: implement')

    def test_ucell_none_allowed(self, backend):
        """ucell may be None at construction and set later (QA scripts do this)."""
        pytest.skip('TODO: implement')

    @pytest.mark.parametrize('bad', ['x', 3, {}])
    def test_invalid_ucell_raises(self, backend, bad):
        pytest.skip('TODO: implement')

    def test_integer_ucell_is_cast_to_float(self, backend):
        """int64/int32 ucell -> type_float. Assert: no dtype error on solve.
        Why: writing `ucell=np.array([[[3, 1]]])` (ints) is the obvious mistake.
        """
        pytest.skip('TODO: implement')

    # _assign_grating_type: 0 = 1D TE/TM, 1 = 1D conical, 2 = 2D.
    @pytest.mark.parametrize('n_rows, pol, phi, fto, expected', [
        (1, 0,   None, [10, 0],  0),   # 1D, TE, no azimuth      -> 1D TE/TM
        (1, 1,   None, [10, 0],  0),   # 1D, TM, no azimuth      -> 1D TE/TM
        (1, 0,   0.0,  [10, 0],  1),   # phi given, even as 0    -> conical
        (1, 0.5, None, [10, 0],  1),   # mixed polarization      -> conical
        (1, 0,   None, [10, 2],  1),   # fto_y != 0              -> conical
        (4, 0,   None, [10, 0],  2),   # 2D pattern              -> 2D
    ])
    def test_grating_type_assignment(self, backend, n_rows, pol, phi, fto, expected):
        """The dispatch table in RCWA*._assign_grating_type.

        Why: phi=0 vs phi=None is the single most surprising branch in the
        package (documented in QA/1d_pattern_in_1dc_and_2d.py). Locking the
        table here means test_equivalence.py can trust which path it exercised.
        """
        pytest.skip('TODO: implement')

    def test_vector_modeling_forces_2d(self, backend, option_2d):
        """Vector modeling currently always assigns grating type 2.
        Note: there is a TODO in the source for a 1D-conical vector path.
        """
        pytest.skip('TODO: implement')


class TestConnectingAlgo:

    def test_tmm_is_default(self, backend):
        pytest.skip('TODO: implement')

    @pytest.mark.parametrize('grating_type', [0, 1, 2])
    def test_smm_raises(self, backend, grating_type):
        """SMM is commented out in _base.solve_* and falls through to ValueError.

        Assert: ValueError on conv_solve with connecting_algo='SMM'.
        Why: pins current reality. When scattering_method.py is wired back up,
        this test flipping to xfail is the signal to write real SMM coverage
        (SMM and TMM must then agree -- that belongs in test_equivalence.py).
        """
        pytest.skip('TODO: implement')

    def test_unknown_algo_raises(self, backend):
        pytest.skip('TODO: implement')
