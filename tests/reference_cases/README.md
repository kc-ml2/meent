# Reference cases — validation against external solvers

One directory per validation experiment. Each holds the numbers an established
code produced, the numbers meent produced, and a small `case.py` saying how to
rebuild the experiment.

`tests/test_reference.py` discovers these automatically. **Adding a case means
adding a directory — no test code changes.**

```
tests/reference_cases/
    _loader.py                      discovery, table loading, capability probes
    README.md                       this file
    <experiment_name>/
        case.py                     how to rebuild it in meent
        README.md                   provenance, measured agreement, caveats
        reference/                  the external solver's output (+ its script)
        recorded/                   meent's output at the time of the experiment
```

Worked example: [`1D_anisotropic_grating_conical_incidence/`](1D_anisotropic_grating_conical_incidence/).

---

## Why both `reference/` and `recorded/`

They answer different questions, and you want both.

| | question | needs |
|---|---|---|
| `reference/` | **Is meent right?** | the external solver (MATLAB, etc.) |
| `recorded/` | **Has meent changed?** | nothing — it is committed data |

The external solver is not part of the test pipeline; RETICOLO needs MATLAB and
nobody will have it in CI. So the experiment's conclusion is preserved as data:
`test_recorded_matches_reference` asserts that meent agreed with RETICOLO at
every wavelength, and it keeps asserting that on any checkout, on any branch,
with no optional dependency installed and no MATLAB anywhere.

The live tests then re-solve the structure with the meent in your working tree
and compare against both.

## What each case gets, for free

Data-only (always run, numpy only):

- reference data conserves energy, has no NaN, covers the declared sweep, and 0 ≤ R, T ≤ 1
- recorded data conserves energy
- **recorded matches reference within tolerance at every wavelength**
- the tolerances in `case.py` still describe the committed files
- `continuous` and `vector` agree with each other

Live (skipped with a specific reason when unavailable):

- live meent matches the reference at the hardest wavelengths
- live meent matches the recorded run (tighter — detects drift below the reference tolerance)
- live meent conserves energy
- the full sweep, under `-m slow`

---

## Adding a case

### 1. Run the experiment

Produce two sets of `lambda_nm, R, T` csv files — one from the external solver,
one from meent. Keep the external solver's script; it goes in `reference/`.

Use the same wavelength grid for both if you can. `_loader.align` matches on the
wavelength column rather than by row, so grids that merely overlap still work.

### 2. Create the directory

```
tests/reference_cases/my_experiment/
    reference/   external output + the script that made it
    recorded/    meent output
```

### 3. Write `case.py`

Copy the one from `1D_anisotropic_grating_conical_incidence/` and edit. The
interface `test_reference.py` relies on:

```python
NAME, DESCRIPTION            # strings
REQUIRES                     # {'backends': [2], 'features': ['anisotropy_diagonal']}
BACKEND                      # which backend the live tests use
METHODS                      # ('discrete', 'continuous', 'vector')
POLS                         # {0: 'TE', 1: 'TM'}
WAVELENGTHS_NM               # the full sweep
SMOKE_WAVELENGTHS_NM         # the handful the default live test uses
TOL_ENERGY                   # |R + T - 1|
TOL_REFERENCE                # {method: tolerance} vs the external solver
MEASURED_REFERENCE           # {method: what the data actually shows}
TOL_METHOD_CROSS             # continuous vs vector
TOL_RECORDED                 # live vs recorded

build(method, pol, wavelength_nm) -> dict     # kwargs for meent.call_mee
observables(result) -> (R, T)
reference_file(pol) -> filename
recorded_file(pol, method) -> filename
```

`case.py` must import with numpy alone — no meent, torch or jax at module level,
or the data-only tests stop working on checkouts without them.

### 4. Measure the tolerances, do not guess them

Compute the actual worst-case deviation from your own files and put it in
`MEASURED_REFERENCE`. Then set `TOL_REFERENCE` about a decade above it.

`test_measured_deviation_is_still_accurate` enforces that the documented
measurement stays true, and `test_tolerances_have_headroom_over_measurements`
catches a tolerance set below what the data already shows. Both exist because a
tolerance picked by loosening until green records nothing.

### 5. Write the case README

Provenance is the part that rots. Record: which external code and **which
version**, the script that ran it, how the meent side was produced (notebook,
branch, commit, how long), the measured agreement, and — most importantly —
**what the case cannot tell you**. The worked example documents that it cannot
detect a φ sign error, because the structure is y-symmetric and only summed
efficiencies are compared. That caveat is worth more than another passing test.

### 6. Declare any feature it needs

If the case needs something not on `main`, add a probe to
`_loader.FEATURE_PROBES` and name it in `REQUIRES['features']`. A probe attempts
the smallest real solve that exercises the feature and returns `(ok, reason)`.

This is what lets reference data be **committed before the feature it validates
is merged**: the data-only tests assert the archived result immediately, and the
live tests announce exactly what is missing instead of failing confusingly.

### 6b. Parking a case whose feature has not merged

A case can be committed before the code it validates exists. The data-only tests
assert the finding immediately; the live tests skip with the probe's reason.

When that happens, add a **`RERUN.md`** next to `case.py` recording what the case
assumes about the unmerged API, what to check when it lands, and what to update
afterwards — see
[`1D_anisotropic_grating_conical_incidence/RERUN.md`](1D_anisotropic_grating_conical_incidence/RERUN.md)
and its companion `verify_after_merge.py`, which checks each API assumption
separately so a changed signature is diagnosed in one line instead of as a wall
of pytest failures.

This matters most when the feature is being written by someone else: the case
records what you needed, so the merge can be checked against it rather than
guessed at.

### 7. Check it

```bash
pytest tests/test_reference.py -v          # data-only + live smoke
pytest tests/test_reference.py -m slow     # the full sweep
```

Then break something on purpose — nudge a number in `recorded/` — and confirm the
right test goes red. A reference test that cannot fail is decoration.

---

## Choosing `SMOKE_WAVELENGTHS_NM`

The full sweep is minutes to hours (the worked example is ~10 minutes per case
at fto = 100), so the default live test runs a handful of points. Pick them where
the recorded run deviates most from the reference, plus an endpoint — not evenly
spaced. Those are the points a regression will move first.

## Things worth knowing

**Energy conservation is necessary, not sufficient.** R and T come from
independent field amplitudes, so R + T = 1 is a real check — but an error that
moves power from R to T passes it untouched. That is what the reference
comparison is for. Both are in the battery for this reason.

**Summed efficiencies are weaker than per-order ones.** Two solvers can agree on
totals while distributing power differently across orders. If you can export
per-order efficiencies from the external solver, do — the comparison gets much
sharper.

**Prefer a structure with no symmetry you are not testing.** A symmetric
structure silently hides sign and handedness errors, as the worked example's
caveat shows.
