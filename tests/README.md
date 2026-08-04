# meent test suite (outline)

**New to pytest? Read [GUIDE.md](GUIDE.md) first** — it is written for physicists
who have never used a test framework, and covers running tests, reading the
output, filling in a stub, and choosing tolerances. This file is the terse
layout reference.

Every test body is currently a `pytest.skip("TODO: ...")` stub. The docstring in
each stub states **Setup / Assert / Why** so the physics decision (what tolerance,
what reference) stays with whoever fills it in.

```
pytest                       # fast unit layer only
pytest -m physics            # invariants: energy, Fresnel, reciprocity, scaling
pytest -m "not slow"         # everything except convergence sweeps and fits
pytest -m regression         # golden values in tests/data/
pytest --regen-golden        # rewrite the golden files (review the diff!)
```

## Layers

The suite is deliberately stratified: the cheap layers must be green before a
failure in the expensive ones means anything.

| File | Layer | Needs a simulation? |
|---|---|---|
| `test_api.py` | dispatch, result containers | no |
| `test_properties.py` | `_BaseRCWA` setters, grating-type assignment | no |
| `test_modeler.py` | raster/vector geometry, nk tables | no |
| `test_fourier.py` | DFS/CFS, convolution matrices | no |
| `test_solver_smoke.py` | shapes, dtypes, determinism | yes (tiny) |
| `test_physics.py` | energy, Fresnel, reciprocity, convergence | yes |
| `test_equivalence.py` | 1D vs 1Dc vs 2D, DFS/EFS/CFS, raster vs vector | yes |
| `test_backend_consistency.py` | numpy vs jax vs torch | yes (×3) |
| `test_field.py` | `calculate_field` | yes |
| `test_autodiff.py` | `grad` / `fit` | yes |
| `test_dtype_device.py` | complex64/128, cpu/gpu | yes |
| `test_regression.py` | golden values from literature cases | yes |
| `test_reference.py` | validation against external solvers | partly — see below |

## Reference cases (the one part that is not a stub)

`tests/reference_cases/` holds real validation experiments: a structure solved
by an external code (RETICOLO), its output, meent's output, and a `case.py`
saying how to rebuild it. `test_reference.py` discovers them and runs the same
battery over each — **adding a case needs no test-code changes**.

The data-only half of that battery asserts the experiment's finding using
committed numbers only, so it runs with numpy alone and passes on branches where
the feature under validation does not even exist. The live half re-solves and
skips with a specific reason when it cannot.

See [reference_cases/README.md](reference_cases/README.md) for the recipe.

## Where this came from

`QA/` already contains the checks this suite formalizes — the scripts print norms
for a human to eyeball; the tests below assert on them:

- `QA/rcwa_backend_consistency.py` → `test_backend_consistency.py` (option dicts
  lifted verbatim into `conftest.py`)
- `QA/1d_pattern_in_1dc_and_2d.py` → `test_equivalence.py::test_1d_matches_1d_conical` / `::test_1d_matches_2d`
- `QA/fourier_analysis_methods.py` → `test_equivalence.py::test_dfs_efs_cfs_agree`
- `QA/autodiff_raster1.py`, `autodiff_raster2.py`, `autodiff_vector.py` → `test_autodiff.py`

The `QA/` scripts are kept as-is (they are useful interactively). The tests are
the machine-checkable subset.

## Tolerances

Tolerance is a physics choice, not a style choice. `conftest.py` exposes a `tol`
fixture keyed on `type_complex`; the defaults there are placeholders. Two notes
carried over from the README:

- complex64 silently loses operations spanning ≳8 digits, so 32-bit tests should
  assert loose agreement or be restricted to well-conditioned cases.
- Eigendecomposition ordering is not stable across backends/versions. Assert on
  gauge-invariant outputs (efficiencies, fields), never on eigenvectors.

## Adding a case

Add the option dict to `conftest.py` (`OPTIONS`) rather than inline in a test —
`test_backend_consistency.py`, `test_physics.py` and `test_regression.py` all
parametrize over that same registry, so one entry buys coverage in all three.
