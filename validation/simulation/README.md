# RCWA simulation cross-validation

This directory validates both the **per-order complex amplitudes** `r_m`, `t_m` and
single-wavelength spatial fields against RETICOLO, rather than only comparing summed
diffraction efficiencies.

There are two material suites, each containing the same twelve geometry/incidence cases,
plus one separate evanescent-input experiment:

| case | status |
|---|---|
| `lossless/1D_isotropic_grating_{normal,oblique,conical}_incidence` | conical **PASS**; smoke runs for the other two were observed but not retained |
| `lossless/1D_anisotropic_grating_{normal,oblique,conical}_incidence` | conical **PASS**; smoke runs for the other two were observed but not retained |
| `lossless/2D_{isotropic,anisotropic}_grating_{normal,oblique,conical}_incidence` | smoke runs observed; artifacts not retained |
| `lossy/{all twelve propagating cases}` | smoke runs observed; artifacts not retained; await RETICOLO references |
| `1D_isotropic_grating_evanescent_input` | **experimental**; see below |

Mounts are `normal` (theta = 0, phi = 0), `oblique` (theta = 30 deg, phi = 0) and `conical`
(theta = 30 deg, phi = 30 deg). The two suites share geometry and angles; the lossy one adds
absorption.

All 24 propagating cases were observed to pass a reduced one-wavelength smoke solve, but those
smoke artifacts were not retained and are not reproducible evidence in this directory. The
complete 201-wavelength meent and RETICOLO output is stored for the lossless 1D isotropic and
anisotropic conical cases; the other cases still need their full meent sweep and `.m` file run in MATLAB.
Until both sides are present, the verdict cell reports incomplete coverage or a missing
reference rather than passing.

Same grating, same angles, same wavelengths, same `fto`. The comparison logic lives once in
[`_compare.py`](_compare.py); each notebook is a config block plus calls.

## Single-wavelength field profiles

Each of the 24 propagating case directories contains exactly one Meent notebook and one
RETICOLO MATLAB script. Both r/t amplitude-coefficient and field-profile workflows live in those two
files. The MATLAB script exposes `RUN_COMPLEX_COEFFICIENTS` and `RUN_FIELD_PROFILE` switches, so a
field-only run does not need to start the 201-wavelength coefficient sweep. The field domain
consists of four finite layers:

```
500e-9 m  superstrate buffer (n_top)
500e-9 m  grating
500e-9 m  slab
500e-9 m  substrate buffer (n_bot)
```

The buffers have exactly the same refractive index as their adjacent semi-infinite media, so
they add no optical interface. They only make the exterior fields available through each
solver's finite-layer field routine.

Workflow for a case:

1. Open `Meent_<case>.ipynb`, set `FIELD_WAVELENGTH_M`, and run the field-settings cell. It
   writes `field_wavelength_m.txt` beside the notebook.
2. In `RETICOLO_<case>.m`, set `RUN_FIELD_PROFILE = true` and, for a field-only run, set
   `RUN_COMPLEX_COEFFICIENTS = false`. Run the script. It reads the wavelength request and writes a
   `RETICOLO_<case>_field_<wavelength>m.mat` reference.
3. Run the final notebook cell. It solves only that wavelength for TE and TM incidence, prints
   relative L2 errors for all six field components in each of the four regions, and plots the
   RETICOLO field, Meent field, and difference.

For 1D cases the plot is the x-z plane. For 2D cases it additionally produces the central y-z
plane and x-y planes at the center of all four layers. Each composite plot contains the
**raw complex field**: `Re` and `Im` of each of `Ex`, `Ey`, `Ez`, `Hx`, `Hy`, `Hz`, with
RETICOLO, Meent, and their normalized absolute difference in adjacent columns. The two
field panels of a row share one symmetric colour scale on a diverging map, so a sign flip
shows up as an inverted colour.

Intensities are deliberately not plotted. `|Ex|^2` discards sign and phase, which is where
a convention error lives - a field off by `-1`, by a conjugation, or by a global phase has
a pixel-identical intensity map. `|E|^2` and `|H|^2` are dropped for the same reason: they
are functions of the six components, so they cannot disagree if those agree. This also
makes the plots show the same quantity the verdict is computed from, since
`field_error_table` has always subtracted the complex arrays directly.

PNG files are saved in that case directory by default. RETICOLO's stored complex field
values are left untouched. Only array axes are reordered to address the same samples.
Meent's public `calculate_field()` output already uses RETICOLO's Cartesian basis and time
convention, so the validation subtracts it directly. No global phase or amplitude is fitted.

Defaults are 81 x-points for 1D, 41 x 33 lateral points for 2D, and 31 z-points per layer.
These settings can be edited in the MATLAB field script; the notebook reads them from the
saved reference so the Meent grid remains identical. `check_grid` then verifies that
RETICOLO's stored `x`, `y` and `z` sample coordinates are the ones Meent will evaluate on,
rather than only checking that the two arrays have the same shape - equal point counts alone
would let a changed sampling rule inside a layer pass as an ordinary numerical error.
Generated `.mat`, request, and optional PNG files are ignored by Git.

At a material edge, the normal electric-field component is discontinuous. RETICOLO and Meent
can therefore reconstruct different values in the Gibbs region around that edge even when the
smooth regions and tangential components agree. The reported table intentionally includes those
samples; the localized edge error is not fitted away or silently excluded, and it carries its
own tolerance, `tol_field_edge`, rather than forcing the tolerance for every other region up
to meet it.

No field samples participate in a phase fit because no phase fit is performed. Edge samples
remain in the error table and are judged only with their separate `tol_field_edge`.

The evanescent-input experiment is excluded: Meent can calculate its internal field, but
RETICOLO `res2`/`res3` does not retain that external incident channel, so it cannot provide a
field reference under the same excitation.

## Notation

Lowercase `r`, `t` are complex **amplitude** coefficients - what this folder compares.
Uppercase `R`, `T` are **power** efficiencies - what `res.de_ri` / `res.de_ti` hold. File
names follow the same rule, so the name alone says which a file holds.

## Why amplitudes

An efficiency-only comparison writes two numbers per wavelength, the summed R and T:

```matlab
if ~isempty(eff{ip,1}); R(ip) = sum(eff{ip,1}); end   % summed over all orders
```

Summing over orders lets per-order errors cancel, and phase is thrown away entirely. Comparing
amplitudes removes both escapes: every order is checked separately, and phase is checked too.

**This does not establish convergence.** Two implementations can agree closely at the same
truncation and both still be short of the exact answer - agreement between codes is not
agreement with the truth. A separate `fto` sweep is what shows convergence, and that is not
what these cases do.

## Normalization, and why the two cases differ

**RETICOLO** normalizes amplitude to energy flux through the xy plane (`res2.m:164`), so

```matlab
'efficacite_TE', abs(amplitude_TE).^2, ...     % res2.m:492
```

**meent** keeps that factor outside (`transfer_method.py:126`). Using `r_raw` below for the
amplitude (the source's local variable is named `R`):

```python
de_ri = (r_raw * r_raw.conj() * (kz_top / (n_top * torch.cos(theta))).real).real
```

so Meent's exported comparison coefficient is `r_m = R_s * sqrt(f)` with
`f = Re(kz / (n_top*cos(theta)))`. RETICOLO's stored coefficient is never rescaled or changed.

That bridge works only while the incident wave propagates. With **evanescent input**
`cos(theta)` is purely imaginary and `f` inverts - zero for exactly the propagating orders,
positive for the evanescent ones:

```
k_par = 0.5 :  f = [0.682, 1.101, 1.000, 0,     0    ]   <- weight on propagating orders
k_par = 1.2 :  f = [0,     0,     1.000, 2.157, 3.153]   <- propagating orders zeroed
```

Multiplying by `sqrt(f)` would zero out every propagating output order, so the evanescent case
runs its meent sweep in `mode='raw'` and writes un-rescaled coefficients for diagnostics.
Those raw coefficients are **not compared directly** with RETICOLO's flux-normalized modal
amplitudes: without incident z-flux there is no demonstrated common normalization, and the
output-mode basis conversion is order dependent. A valid cross-check must instead normalize
physical E/H from both codes to the same nonzero incident-field component. Until that mapping
is implemented, `_compare.py` deliberately skips the comparison and the verdict fails.

Efficiency is meaningless there for the same reason: no incident z-flux means a zero
denominator, and meent reports `R + T = 4.17`. That is not a solver error.

The self-check is nevertheless always run on the **flux-normalized** amplitudes, in both modes:
`sum|r_raw*sqrt(f)|^2 == sum|r_raw|^2*f == de` is an identity, while `sum|r_raw|^2` alone is not. Checking
the raw amplitudes against `de` would report a meaningless mismatch and lose what the check is
for - confirming that the `kz` branch selection in `_compare.py` still matches the solver's.

Other conventions, handled in the files here:

* **delta sign** - meent `phi = +X` corresponds to RETICOLO `angle_delta = -X`.
* **s/p naming** - meent `_s` is RETICOLO `_TE`; meent `_p` is RETICOLO `_TM`.

## How the complex coefficient is compared

Meent is exported in RETICOLO's normalization, order-local polarization basis, time
convention, and reference plane. The complex values are then subtracted directly:

```
err = max |meent_vector - ret_vector|
```

The vector holds **every propagating order, both polarization components, and both directions
(r and t)**. No phase, amplitude, sign, or conjugation is fitted:

* **TE and TM together** - a relative phase error between polarization components remains
  visible.
* **r and t together** - a constant phase error on every `t_m` while `r` stays correct also
  remains visible.

A conjugated Meent vector is measured only as a diagnostic for a possible time-convention
regression; it is never substituted into the PASS calculation.

A different **reference plane** is not removed either, and must not be assumed harmless.
Moving the plane by `dz` multiplies order `m` by `exp(i*kz_m*dz)`, and `kz_m` differs per
order. Direct comparison exposes both common and order-dependent reference-plane phase errors.

Because the error is on the direct complex amplitude rather than on an angle, the tolerance
is named `tol_complex`, not `tol_phase`.

## Verdict

The propagating notebook ends in `C.verdict(...)`, which **raises** on any failure. A run that
finishes without an exception is the pass signal; printed mismatches cannot look green. The
evanescent notebook currently raises intentionally because its cross-code field normalization
is not implemented. The verdict checks:

* RETICOLO reference present **and non-empty** (a header-only file would make every expected
  count zero and pass vacuously)
* no failed meent solves
* exact meent table coverage for the configured wavelength and order grid, even before a
  RETICOLO reference exists (so smoke output cannot masquerade as a full result)
* a sweep manifest matching `fto`, wavelength grid, geometry and complex materials, so a
  lower-order run with the same output keys cannot overwrite a production result silently
* no silently skipped orders or wavelengths, reported by cause
* normalization self-check within `tol_selfcheck`
* **coverage** - the comparisons that should have run, ran. The loops skip past empty files,
  NaN coefficients and mismatched order sets; without an explicit count those look exactly
  like "compared and agreed"
* magnitude and direct complex amplitude within per-method tolerance

The field cell is judged the same way. `compare_field_profile` prints its table, draws its
plots, and then runs `F.field_verdict(...)`, which **raises**; the plots are produced first so
a failing run still leaves the figures that show where it failed. Pass
`raise_on_failure=False` to get the result dict back from a run that would otherwise raise.
It checks:

* the error table covers every (polarization, region, component) entry, so a table that lost
  entries cannot pass by having nothing left to check
* no non-finite entry
* every entry within `tol_field`, except the electric component normal to the grating edge,
  which is held to `tol_field_edge`

Both field tolerances are **provisional**. They are sized from the one case whose field run is
recorded below, not from a sweep over all 24.

## Layout

```
_compare.py                  all comparison logic, config-driven
_field_compare.py            single-wavelength field construction, alignment, errors, plots
reticolo_export_field.m      shared RETICOLO res3 field exporter
reticolo_setup.m             shared RETICOLO path and MATLAB output-directory setup
lossless/<case>/             twelve lossless cases
lossy/<case>/                the same twelve cases with absorbing materials
  RETICOLO_<case>.m          run in MATLAB -> RETICOLO_<case>_{TE,TM}_{r,t}.txt
                              and/or RETICOLO_<case>_field_<wavelength>m.mat
  Meent_<case>.ipynb         r/t amplitude comparison, field comparison, and saved field-profile PNGs
  Meent_<case>_manifest.txt  exact configuration that generated the meent tables
1D_isotropic_grating_evanescent_input (X)/
                              unsupported external-channel experiment
```

The MATLAB scripts use `RETICOLO_ROOT` (the directory containing `res1.m`) when it is set.
Otherwise they look for a sibling
checkout at `<parent>/reticolo/reticolo`, matching the common layout where the `meent` and
`reticolo` repositories share a parent directory. Generated references are always written
beside the case notebook, regardless of MATLAB's starting directory.

File format, one row per diffracted order per wavelength:

```
wavelength_m, order_x, order_y, te_re, te_im, tm_re, tm_im, efficiency
```

The evanescent case's `.m` appends twelve columns holding output `E` and `H` at the origin.
They are useful ingredients for a future field comparison, but are still produced from
RETICOLO's normalized modal basis and scattering amplitudes. They are not, by themselves, an
incident-normalization-independent fallback. The comparison code therefore ignores these
columns until the corresponding incident field and meent field-basis mapping are implemented.

Notebooks are stored without execution output, so diffs stay readable. Consequently, the
repository does not claim that a cleared notebook is itself evidence of a completed run. A
validation result is established by rerunning through the verdict cell; the `Results` section
below records the latest observed result and the saved coefficient files make the numerical
comparison reproducible.

The setup cell locates the repository by walking upward from the kernel's current directory,
so the notebooks can be launched from the repository root, `validation/simulation`, or an
individual case directory.

## Results

**Lossless propagating input, 1D isotropic conical** - full run, 201 wavelengths, `fto=100`,
639 (wavelength, order) pairs per direction:

| quantity | discrete | continuous | vector |
|---|---|---|---|
| max `d\|a\|` | 1.1e-05 | 1.1e-10 | 3.4e-11 |
| max direct complex error | 1.2e-03 | 1.5e-10 | 4.1e-11 |

Self-check max `1.443e-15` over 2412 combinations. Time convention matched as-is at 100% of
wavelengths. The `discrete` column is raster discretization, not disagreement, which is why
tolerances are per method.

**Lossless propagating input, 1D anisotropic conical** - full run, 201 wavelengths, `fto=100`,
639 (wavelength, order) pairs per direction:

| quantity | discrete | continuous | vector |
|---|---|---|---|
| max `d\|a\|` | 1.4e-05 | 1.1e-10 | 1.1e-10 |
| max direct complex error | 1.4e-03 | 1.4e-10 | 1.3e-10 |

Self-check max `1.332e-15` over 2412 combinations. Time convention matched as-is at 100% of
wavelengths. Its `600e-9 m` continuous-field comparison has errors at or below `8.591e-12`
outside the discontinuous normal electric component at the grating edge; the full grating-region
`Ex` errors are `4.569e-02` for TE and `6.327e-02` for TM. Those are direct complex
differences - no phase was removed, because `compare_field_profile` fits none.

An earlier revision reported `1.101e-06` for that same non-edge figure. That number was an
artifact: the global phase was fitted over all samples including the grating-`Ex` block, and
the resulting ~1e-6 rad phase was then applied to regions where the two codes agree to
`3e-12`. Dropping the fit removed the artifact; the fields themselves did not change.

**Lossy propagating input** - all twelve configurations were observed to pass a one-wavelength
reduced-order smoke solve, with `R + T` below one as expected for passive media. Neither the
smoke outputs nor manifests were retained, so this is an observation rather than reproducible
validation evidence. No lossy RETICOLO coefficient references are stored yet.

**Evanescent input** - the meent sweep runs clean (0 failed solves, self-check `1.066e-14`),
but this is not a cross-solver PASS. RETICOLO was run over all 201 wavelengths without a
solver error, but `res2` returned empty per-order blocks, so no usable RETICOLO coefficient
reference files were produced. Direct raw-amplitude comparison remains disabled even if an
evanescent channel is forced into the output, because the two amplitudes do not yet share a
demonstrated normalization. The remaining work is to export a usable incident E/H reference,
reconstruct meent's per-order E/H in the same coordinates, and divide both by the same
nonzero incident-field component.

A prior ad-hoc Fresnel-Airy check of a uniform slab gave unit magnitude ratio and zero phase
spread at `k_parallel = 1.2` and `1.8`, but its calculation is not currently stored here.
Treat it as an observation, not reproducible validation evidence, until a script or notebook
cell is added.
