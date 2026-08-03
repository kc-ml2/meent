"""Public entry point: meent.call_mee and the result containers.

Cheapest layer in the suite. If these fail, nothing below them means anything.
"""

import pytest


class TestCallMee:

    def test_returns_expected_class(self, backend):
        """call_mee(backend=n) -> Mee{Numpy,Jax,Torch}.

        Assert: type(mee).__name__ matches the backend, and `mee.backend == n`.
        Why: main.py dispatches on a bare int; a reordering there is silent.
        """
        pytest.skip('TODO: implement')

    @pytest.mark.parametrize('bad', [3, -1, 'numpy', None])
    def test_invalid_backend_raises(self, bad):
        """Assert: ValueError. Why: pins the contract that it fails loudly."""
        pytest.skip('TODO: implement')

    def test_kwargs_reach_both_parents(self, backend):
        """Mee* multiply-inherits Modeling* and RCWA* (and Optimizer* for 1/2).

        Assert: an option (e.g. type_complex, period) set via call_mee is visible
        on the instance, for a kwarg consumed by each parent's __init__.
        Why: MeeTorch.__init__ forwards *args/**kwargs to three __init__s; a
        signature change in one parent breaks construction only for some kwargs.
        """
        pytest.skip('TODO: implement')

    @pytest.mark.jax
    def test_jax_x64_enabled_on_import(self):
        """meent/__init__.py sets jax_enable_x64 inside a bare try/except.

        Assert: after `import meent`, jax.config.jax_enable_x64 is True.
        Why: silently falling back to float32 would degrade every jax result
        without raising -- exactly the failure the except: swallows.
        """
        pytest.skip('TODO: implement')


class TestResultContainers:

    def test_result_exposes_three_channels(self, backend, option_2d):
        """conv_solve() -> Result* with .res, .res_te_inc, .res_tm_inc."""
        pytest.skip('TODO: implement')

    def test_de_ri_de_ti_proxy_to_res(self, backend, option_2d):
        """Result.de_ri is res.de_ri. Assert: identical arrays."""
        pytest.skip('TODO: implement')

    def test_proxies_return_none_when_res_absent(self):
        """Result*(res=None).de_ri is None, not an AttributeError."""
        pytest.skip('TODO: implement')

    def test_te_tm_channels_none_for_1d_tetm(self, backend, option_1d_te):
        """transfer_1d_4 returns only {'res': ...}; conical/2D return all three.

        Assert: grating type 0 -> res_te_inc is None and res_tm_inc is None.
        Why: documents the asymmetry so a caller-side `.res_te_inc.de_ri` on a
        1D TE/TM case fails here rather than in user code.
        """
        pytest.skip('TODO: implement')

    def test_sub_result_has_all_attrs(self, backend, option):
        """Assert: every name in conftest.RESULT_ATTRS is present and finite."""
        pytest.skip('TODO: implement')

    def test_expected_methods_exist(self, backend):
        """conv_solve, calculate_field, conv_solve_field, field_plot on all three;
        grad/fit only on jax and torch.
        """
        pytest.skip('TODO: implement')
