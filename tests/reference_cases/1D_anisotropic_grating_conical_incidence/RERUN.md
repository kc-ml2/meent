# Re-running this case after a change to the solver

> **Status: live and passing.** Anisotropy merged in
> [#103](https://github.com/kc-ml2/meent/pull/103) (`58ff99f`) and this case was
> re-run against it on 2026-08-04 — all 8 contract checks pass, 104 tests pass,
> nothing skipped for missing features. Results in
> [What the merge run found](#what-the-merge-run-found) below.
>
> The procedure below stays useful for the *next* solver change: run
> `verify_after_merge.py` first, `pytest` second.

The value of running the script before pytest is that it checks each API
assumption separately. A changed `ucell` signature otherwise surfaces as dozens
of identical pytest failures that say nothing about which interface moved.

---

## TL;DR

```bash
git checkout main && git pull                    # the merged anisotropy
git checkout feature/pytest && git rebase main   # bring the tests onto it
pip install -e . && pip install pytest torch

python tests/reference_cases/1D_anisotropic_grating_conical_incidence/verify_after_merge.py
```

**Check the first line it prints.** It reports which `meent` was imported. If
that path is a `site-packages` copy rather than your working tree, you are
testing the wrong code — `pip install -e .` from the repo root and re-run. This
is easy to miss and makes every result meaningless.

Read its output, apply what it tells you, then:

```bash
pytest tests/test_reference.py -v                # data-only + live smoke
pytest tests/test_reference.py -m slow           # full 201-point sweep, ~10 min
```

---

## What the merge run found

Run on 2026-08-04 against `main` @ `58ff99f`, Linux, torch 2.13.0+cpu,
numpy 2.4.6, meent 0.13.2 editable from the working tree.

| | |
|---|---|
| contract checks | **8/8 pass** — the merged API matches what `case.py` assumed |
| `pytest tests/test_reference.py` | **104 passed**, 6 deselected (`slow`) |
| data-only | 32 pass |
| live | 72 pass — previously skipped |

Numbers, all inside their asserted tolerances:

| comparison | discrete | continuous | vector |
|---|---|---|---|
| vs RETICOLO | 1.03e-05 | 6.15e-11 | 6.42e-11 |
| vs the recorded run | 4.65e-12 | 3.09e-12 | 4.94e-12 |

Worst energy residual 3.53e-12 against a 1e-9 tolerance.

**The headline is the middle row.** The merge included a unified iso/aniso solver
path (`dc4b077`) and a fix to kz branch selection for evanescent orders
(`1c3017b`) — a refactor and a physics change, either of which could have moved
this case. Live meent reproduces the Windows recorded sweep to **5e-12**, on a
different OS and a different LAPACK build. The refactor preserved behaviour.

That also finally measured `TOL_RECORDED`, which had been a guess: it is now
1e-10, tightened from 1e-9.

Two smaller findings from the run:

- `pol=0`'s `.res` is **identical** to `res_te_inc` from the same solve (to 12
  decimals). The sweep can be halved — solve once per wavelength and read both
  polarization channels — if the full sweep's ~30 minutes ever becomes annoying.
- Install trap: `MeeTorch` pulls in `meent/on_torch/optimizer/optimizer.py`,
  which imports `tqdm` — so `call_mee(backend=2)` raises `ModuleNotFoundError`
  before solving anything. `tqdm` is correctly declared in the `pytorch` extra,
  so the packaging is right; what is wrong is installing `-e .` and `torch`
  separately. Use `pip install -e ".[pytorch]"`.

---

## The API contract

`case.py` encodes eight assumptions about how anisotropy is used. Each has a
matching check in `verify_after_merge.py`. This table is the thing to consult
when a check fails.

| # | Assumption | Encoded in | If the PR changed it |
|---|---|---|---|
| 1 | ucell of shape `(layers, H, W, 3)` means a diagonal (nx, ny, nz) tensor | `case.build_raster` | Rewrite `build_raster` to the new spelling (e.g. an explicit flag or a separate argument). Keep the *structure* identical — 4 columns of 250 nm, bar spanning 250–750 nm. |
| 2 | An anisotropic conical case solves at all | `case.build` | If it raises, the case cannot run; report it as a PR bug rather than adapting. |
| 3 | Vector instructions accept a 3-component index | `case.build_vector` | Adapt the instruction format. If vector anisotropy did not land, drop `'vector'` from `case.METHODS` and say so here — the discrete/continuous comparison still stands. |
| 4 | Anisotropic + conical routes to the 2D solver | `case.py` docstring | Only a comment is wrong, unless the routing changed the *answer*. Check the numbers in section 2 of the script before editing anything. |
| 5 | `type_complex=0` means complex128 | `case.TYPE_COMPLEX` | Update. Note that a silent drop to complex64 would move the deviations from 1e-11 to ~1e-6 and look like a physics regression. |
| 6 | The isotropic path still works | — | This is a guard on the PR, not on the case. A failure here is a merge defect worth raising immediately. |
| 7 | A conical solve exposes `res_te_inc` / `res_tm_inc` | not used by `case.py` | Informational. |
| 8 | `pol=0`'s `.res` equals `res_te_inc` | not used by `case.py` | Informational — if it holds, the sweep can be halved by solving once per wavelength instead of once per polarization. Worth doing before the full sweep. |

## If a contract check fails

**Do not widen a tolerance to make anything pass.** The tolerances here were
measured from data, and the whole point of the case is that they are falsifiable.

Work in this order:

1. **Contract failure (section 1 of the script)** — an interface moved. Fix
   `case.py` to speak the new interface, keeping the physical structure
   identical. Re-run the script. Nothing in section 2 means anything until
   section 1 is clean.
2. **Energy conservation fails but RETICOLO agreement holds** — unlikely
   combination; suspect the observable extraction (`case.observables`) rather
   than the solver.
3. **RETICOLO agreement fails** — this is the interesting one. See below.

## If the numbers disagree

The question is always *which side moved*. Three sources of evidence, in order
of cost:

1. **Is it all methods or one?** `discrete` alone drifting is a Fourier-expansion
   change. All three moving together is the solver or the structure.
2. **Is it all wavelengths or a few?** The script prints per-wavelength
   deviations. A single bad wavelength near a resonance is a conditioning
   problem; a uniform offset is a normalization change.
3. **Does energy still conserve?** If R + T = 1 holds but RETICOLO disagrees,
   power is being moved between R and T — a normalization or branch-cut sign,
   not a bookkeeping error. We verified this failure mode is invisible to the
   energy check by mutating the data, which is why both tests exist.

If meent legitimately improved (say `discrete` now agrees to 1e-8), that is a
**finding, not a chore**: tighten `TOL_REFERENCE`, update `MEASURED_REFERENCE`
and the table in README.md, and say why in the commit message. Do not leave a
loose tolerance in place because it passes.

## What to update after a good run

The script prints these at the end, ready to paste:

- **`TOL_RECORDED`** in `case.py` — currently a guess (1e-9). This is the number
  that has been waiting for a machine that can actually run the case. The
  recorded sweep ran on Windows with a different LAPACK build and
  eigendecomposition is not bit-reproducible across builds, so expect ~1e-11,
  not 0.
- **`MEASURED_REFERENCE`** — only if you ran `-m slow`. The script's figures come
  from the 4-wavelength smoke subset and are not the full-sweep worst case.
- **README.md** in this directory — the measured-agreement tables, and the
  RETICOLO version, which is still unrecorded.
- **This file** — delete the "parked" framing once the case runs.

---

## Inventory

**Committed here** — self-contained; re-running or re-auditing the experiment
needs nothing outside this directory:

```
reference/  RETICOLO_..._TE.txt, _TM.txt        201 wavelengths each
            RETICOLO_....m                      the MATLAB script that made them
recorded/   Meent_..._{TE,TM}_{discrete,continuous,vector}.txt
Meent_1D_anisotropic_grating_conical_incidence.ipynb
            the notebook that produced recorded/, with its stored outputs
case.py     structure, sweep, tolerances, meent construction
README.md   provenance, measured agreement, caveats
RERUN.md    this file
verify_after_merge.py
```

**Left in the source folder** (`/home/chs/Work/Meent/meent/1D_anisotropic_grating_conical_incidence/`),
deliberately: `prv1.mat` (16 MB RETICOLO intermediate), `.xlsx` (the `.txt`
numbers reorganized for plotting), `.opju` (Origin project). Derived or
regenerable, and large.

> That folder was briefly emptied on 2026-08-04 when a second checkout switched
> branches — the files were untracked, so nothing in git held them. It has been
> restored, and the nine committed data files were verified byte-identical to the
> restored originals. The notebook is now committed here rather than living only
> in an untracked folder. Worth remembering when the next experiment is run:
> **untracked is not kept.**

**What the notebook adds beyond `case.py`.** `case.py` reproduces the solve path
exactly (same structure, same fto, same sweep) and the notebook's eyeball checks
are now assertions in `tests/test_reference.py`. The notebook is still the only
record of one investigation: a **convolution-matrix stability probe** that blends
the discrete and continuous matrices as `C(t) = A + t(B − A)` and compares
`t = 1 − 1e-12` against `t = 1`. Those two matrices are numerically identical, so
any change in R between them is a solver breakdown rather than a physics
difference. It found nothing above 1e-4 in this dataset. Worth promoting to a
test if that failure mode matters — it is a genuinely good idea and it is not
covered anywhere else in the suite.

---

## Reference numbers

Recomputed from the committed files. `verify_after_merge.py` compares against
these; they are also in README.md.

meent vs RETICOLO, max(|ΔR|, |ΔT|) over 201 wavelengths:

| method | TE | TM | asserted |
|---|---|---|---|
| discrete | 6.25e-06 | 1.03e-05 | 1e-4 |
| continuous | 5.06e-11 | 6.20e-11 | 1e-8 |
| vector | 5.06e-11 | 6.24e-11 | 1e-6 |

Energy, max |R + T − 1|: RETICOLO 9.81e-11; meent 9.76e-12 (discrete),
1.14e-11 (continuous), 2.93e-11 (vector). Continuous vs vector: 1.05e-11.

Smoke wavelengths and why: **540 nm** worst TM discrete, **565 nm** worst TE
discrete, **639 nm** worst continuous/vector and worst energy residual,
**700 nm** endpoint.

---

## Also worth checking at merge time

`tests/test_properties.py::TestGratingTypeAssignment` is a stub whose expected
table was written from the numpy backend. On the anisotropy branch, torch's
`_assign_grating_type` treats `phi=0` the same as `phi=None` (routing to the fast
1D solver), whereas numpy and jax deliberately let `phi=0` force the conical
path — see `QA/1d_pattern_in_1dc_and_2d.py` for that contract. If the merged PR
keeps that difference, the stub needs per-backend expectations, and the
divergence deserves a note in the main README.
