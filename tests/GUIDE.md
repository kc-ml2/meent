# A guide to testing meent

Written for the people who write the physics, not for software engineers. It
assumes you know RCWA and have never used pytest. Nothing below is about
software craftsmanship for its own sake — the goal is narrow: **catch a wrong
number before a user builds a paper on it.**

---

## 1. Why bother — you are already doing this

Look at `QA/rcwa_backend_consistency.py`. It runs the same structure through
NumPy, JAX and PyTorch, computes the norm of the differences, and prints them.
Someone then reads the numbers and decides whether they look small enough.

That is a test. It has exactly two problems:

1. **It needs a human.** Nobody runs it on a Tuesday afternoon after a small
   refactor. It gets run when someone already suspects something.
2. **"Looks small enough" is not recorded.** The threshold lives in the head of
   whoever last looked at the output.

pytest fixes both. You write the comparison once, you state the tolerance once
in code, and from then on a machine checks it on every change and shouts if it
breaks. The physics content is identical — you are just writing down the
judgment you were making by eye anyway.

The payoff is specific to numerical work: **a bug in an RCWA solver usually does
not crash.** It returns a plausible number. A missing `cos(theta)` in the flux
normalization gives you efficiencies that are smooth, positive, and wrong by 3%.
You will not spot that by looking at a spectrum. You will spot it instantly if
something checks that energy sums to 1.

---

## 2. Setup

```bash
pip install pytest
pip install -e .          # install meent in editable mode, from the repo root
```

Editable mode (`-e`) means the tests import the code in your working directory,
so your edits take effect immediately without reinstalling.

Optional backends, as needed:

```bash
pip install "meent[jax]"      # or: pip install jax jaxlib optax
pip install "meent[pytorch]"  # or: pip install torch
```

Tests for a backend you have not installed are skipped automatically — you never
need to comment anything out.

---

## 3. Running tests

From the repo root:

```bash
pytest                          # everything fast (this is what you run most)
pytest tests/test_physics.py    # one file
pytest tests/test_physics.py::TestEnergyConservation   # one class
pytest -k "energy"              # every test whose name contains "energy"
pytest -m physics               # every test tagged as a physical invariant
pytest -m "not slow"            # skip the long convergence sweeps
pytest -x                       # stop at the first failure
pytest -v                       # one line per test instead of one character
pytest --lf                     # re-run only what failed last time
```

`-k` and `--lf` are the two you will use constantly while fixing something.

### Reading the output

```
tests/test_physics.py ..s.F                                              [ 45%]
```

One character per test:

| | |
|---|---|
| `.` | passed |
| `F` | **failed** — an assertion was false. This is a finding. |
| `E` | **error** — the test crashed before it could assert. Usually a typo or a missing fixture, not physics. |
| `s` | skipped — backend missing, marked slow, or still a `TODO` stub |
| `x` | expected failure (`xfail`) — a known-broken thing, recorded on purpose |

On failure pytest prints the values involved:

```
E   AssertionError:
E   Not equal to tolerance rtol=0, atol=1e-08
E   Mismatched elements: 1 / 1 (100%)
E   Max absolute difference: 0.0031
E    x: array(0.9969)
E    y: array(1.0)
```

That is the whole diagnostic: total energy came out 0.9969 instead of 1. No
debugger needed.

---

## 4. What a test actually is

A function whose name starts with `test_`, that computes something and asserts
a claim about it. That is the entire concept.

```python
def test_pol_zero_is_pure_te():
    mee = meent.call_mee(backend=0, pol=0)
    assert mee.psi == pytest.approx(np.pi / 2)
```

pytest finds every `test_*` function in every `test_*.py` file, runs it, and
reports which assertions held. If the function returns without raising, it
passed.

**Never compare floats with `==`.** Use one of these:

```python
assert value == pytest.approx(expected, rel=1e-9)     # scalars
np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)   # arrays
```

`assert_allclose` checks `|actual - expected| <= atol + rtol * |expected|`. Set
`rtol=0` when you want a pure absolute check (energy conservation), and `atol=0`
when you want a pure relative one (comparing two efficiencies of similar size).
Mixing them carelessly is the most common way to write a test that silently
passes on garbage: with a default `atol` of `1e-8`, any comparison of numbers
near zero succeeds no matter what.

---

## 5. Filling in a stub

Every test in this suite currently looks like this:

```python
def test_pol_sets_psi(self, backend, pol, expected_psi):
    """psi = pi/2 * (1 - pol). Assert: mee.psi == expected."""
    pytest.skip('TODO: implement')
```

The docstring says what to do. You delete the `pytest.skip` line and write it.

### Worked example 1 — the mechanical kind

```python
@pytest.mark.parametrize('pol, expected_psi', [
    (0, np.pi / 2),
    (1, 0.0),
    (0.5, np.pi / 4),
])
def test_pol_sets_psi(self, backend, mee, pol, expected_psi):
    instance = mee(backend, pol=pol)
    psi = float(np.real(to_numpy(instance.psi)))
    assert psi == pytest.approx(expected_psi)
```

`@parametrize` runs this three times, once per row, and reports them as three
separate tests. If only `pol=0.5` breaks, you see exactly that. It is much
better than a loop inside one test, which stops at the first failure and tells
you nothing about the rest.

### Worked example 2 — the kind that matters

```python
def test_lossless_structure_conserves_energy(self, backend, mee, option_1d_te):
    instance = mee(backend, **option_1d_te)
    result = instance.conv_solve()

    de_ri = to_numpy(result.res.de_ri)
    de_ti = to_numpy(result.res.de_ti)
    total = de_ri.sum() + de_ti.sum()

    np.testing.assert_allclose(total, 1.0, rtol=0, atol=1e-8)
```

Six lines, and it covers the flux normalization, the transfer-matrix assembly,
the incident-field construction and the Fourier expansion simultaneously.

**But it is not that simple, and the subtlety is pure physics.** `option_1d_te`
uses `fto=0` — only the zeroth order is retained. Energy will *not* sum to 1,
because the truncated orders carry real power away. The test as written above
would fail, and it would be right to fail.

So the test needs a converged `fto`, and "converged" depends on the structure.
That decision is yours, not pytest's. Write it down in the test:

```python
    option = dict(option_1d_te, fto=40)   # converged for this structure; see
                                          # test_physics.py::test_efficiency_converges_with_fto
```

This is the general shape of scientific testing: **the tooling is trivial, the
physics judgment is the work.** Do not let a failing test push you into
loosening a tolerance until it goes green. Either the number is right and you
can justify the tolerance, or something is wrong and you have just found it.

---

## 6. The three pytest concepts you need

### Fixtures — shared setup

A fixture is a named piece of setup that a test can ask for by putting its name
in the function signature. They live in `conftest.py`.

```python
def test_something(backend, option_1d_te):   # pytest supplies both
    ...
```

- `backend` — runs the test once per installed backend (0, 1, 2).
- `option` — runs it once per canonical case in `conftest.OPTIONS`.
- `option_1d_te`, `option_2d` — one specific case.
- `mee` — `mee(backend, **option)` builds a configured instance.
- `solve` — `solve(backend, **option)` builds one *and* runs `conv_solve`.
- `tol` — `tol(type_complex)` gives the agreed `rtol`/`atol` for that precision.

Asking for `backend` and `option` together gives you 3 × 5 = 15 runs from one
function body. That is where the leverage is.

**Add new physical cases to `conftest.OPTIONS`, not inline in a test.** Three
different files parametrize over that registry, so one entry buys coverage in
all of them.

### Parametrize — the same test, many inputs

Shown above. Use it for angle sweeps, polarizations, dtypes, `fto` values —
anything where you would otherwise write a loop or copy-paste the test.

### Markers — labels for selecting subsets

```python
@pytest.mark.slow       # excluded by `pytest -m "not slow"`
@pytest.mark.physics    # an invariant, not an implementation detail
@pytest.mark.jax        # needs jax
```

The full list is in `pytest.ini`. Tag anything that takes more than a couple of
seconds as `slow` — if the default `pytest` run stops being fast, people stop
running it, and then it protects nothing.

---

## 7. Rules of thumb for numerical tests

**Assert on gauge-invariant quantities.** Efficiencies, fields, energies — yes.
Eigenvectors — no. Eigendecomposition ordering and phase are not stable across
LAPACK builds, let alone across NumPy/JAX/PyTorch. A test on eigenvectors will
fail on someone else's laptop for no physical reason.

**Prefer invariants over stored numbers.** "Energy sums to 1", "scaling λ and
the period together changes nothing", "a uniform layer reproduces Fresnel" —
these need no reference data, never go stale, and tell you *what* broke. Golden
files tell you only *that* something changed.

**Get one absolute check per formulation.** Everything else in this suite is
self-consistency: it proves the code agrees with itself. A closed form (Fresnel
for a uniform slab, the analytic sinc coefficients of a binary grating) or a
published table is what proves it was right to begin with.

**A test that has never failed has proven nothing.** After writing one, break
the code on purpose — flip a sign, drop a `cos(theta)` — and confirm it goes
red. Then undo. This takes thirty seconds and is the only way to know the test
is wired up to anything.

**Never use unseeded randomness.** `np.random.rand(...)` gives a different
structure every run, so a failure cannot be reproduced. Use
`np.random.default_rng(0)`.

**Keep the default run fast.** Small `fto`, few layers, coarse field grids. Save
the expensive cases for `-m slow`.

**Don't test the implementation, test the claim.** Asserting that an internal
matrix has a particular shape locks in today's code and blocks refactoring.
Asserting that the transmitted power is correct does not.

---

## 8. When a test fails

Work up the stack, cheapest first — that is why the suite is layered:

1. Did `test_properties.py` or `test_fourier.py` also fail? Then the problem is
   in input handling or the convolution matrices, and every physics failure
   downstream is just an echo. Fix that first.
2. Did only one backend fail? Compare against `test_backend_consistency.py` —
   it isolates whether the divergence is in the modeler or the linear algebra.
3. Did energy conservation fail but nothing else? Suspect the flux
   normalization in `transfer_*_4`, or an unconverged `fto` in the test itself.

**A failing regression test is a finding, not a chore.** If `tests/data/` no
longer matches, some number changed. Find out which and why *before* running
`--regen-golden`. Regenerating a golden file to silence a red test discards the
exact information the test existed to give you. When the change is intentional,
regenerate it in its own commit with the physics reason in the message.

---

## 9. Cheat sheet

```bash
pytest                              # fast suite
pytest -x -k energy                 # first failing energy test, stop there
pytest -v tests/test_physics.py     # verbose, one file
pytest -m "physics and not slow"    # invariants, quick ones only
pytest --lf                         # only what failed last run
pytest -m regression                # golden values
pytest --regen-golden               # rewrite goldens (fails on purpose; review the diff)
```

```python
assert x == pytest.approx(y, rel=1e-9)              # scalar
np.testing.assert_allclose(a, b, rtol=0, atol=1e-8) # array, absolute
with pytest.raises(ValueError):                     # assert something is rejected
    meent.call_mee(backend=99)
pytest.skip('not implemented yet')                  # placeholder
@pytest.mark.parametrize('n', [1, 2, 5])            # same test, three inputs
@pytest.mark.slow                                   # takes a while
```

---

## 10. Where to start

The stubs are ordered by value per unit effort. In rough priority:

1. `test_physics.py::TestEnergyConservation` — the single highest-value test here.
2. `test_physics.py::test_uniform_layer_matches_fresnel` — your one absolute
   reference; everything else is self-consistency.
3. `test_equivalence.py` — the QA scripts already compute these norms, so it is
   mostly a matter of choosing thresholds.
4. `test_properties.py` — tedious but fast, and it catches the input-coercion
   bugs that masquerade as physics bugs.
5. Everything else.

Five stubs are marked `TODO: decide ...` rather than `TODO: implement`:

```bash
grep -rn "TODO: decide" tests/
```

Those are open questions about what meent *should* do — what happens when
`len(thickness)` disagrees with the number of ucell layers, what `find_nk_index`
does off the end of a table, what `calculate_field` does before `conv_solve`,
whether `set_field_input` applies to the 1D TE/TM path, and which exact
reciprocity relation to assert. They need a decision from the physics side
before anyone can write the assertion — please settle them rather than deleting
them. A sixth open item, `test_backend_consistency.py::test_tolerance_is_documented`,
asks for the backend agreement tolerances to be *measured* on a known-good
commit rather than guessed.
