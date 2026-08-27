# nk_data

Refractive index tables read by `read_material_table()`. Any `.txt` or `.mat` file below this
directory is picked up; the **file name is the lookup key**, case-insensitive, and the directory
it sits in is ignored.

```python
import meent

mee = meent.call_mee(backend=2, wavelength=1.55e-6, period=[1.0e-6], thickness=[2.0e-7])
mee.ucell = mee.put_refractive_index_in_ucell(ucell, ['sio2_malitson', 'au_johnson'], 1.55e-6)
```

Use `meent.print_materials()` to list everything available with its wavelength range, or
`meent.list_materials()` for the same data as dicts. To read one index directly, without going
through a solver:

```python
table = meent.read_material_table()
meent.find_nk_index('au_johnson', table, 1.55e-6)   # 0.5241 - 10.7424j
```

Each backend carries its own copy of these two functions under `meent.on_*/modeler/modeling.py`;
they return the same values, and the ones re-exported here are the numpy set.

## Sign convention

Tables store `k` as the positive extinction coefficient, the way every source publishes it.
`find_nk_index` returns `n - ik`, because the solver runs on the negative sign convention. Handed
`n + ik` a material amplifies instead of absorbing: transmission through a slab grows with
thickness rather than decaying, and `R + T` exceeds 1. That is the quickest check that an index
has the wrong sign.

All three backends return the same sign, so no `.conj()` is needed anywhere:

```python
mee.ucell = mee.put_refractive_index_in_ucell(ucell, ['sio2_malitson', 'au_johnson'], wl)
```

The `__real` suffix drops `k` entirely and returns `n` on its own.

## Wavelengths are in metres

meent itself is unit-agnostic -- only the ratio of wavelength to period matters -- so these
tables are the one place an absolute unit is fixed. **Every folder here is in metres**, in the
tables and in the `# range:` headers alike, so `find_nk_index` always wants a wavelength in
metres.

Interpolation clamps to the endpoints outside a table's range, so a wavelength in the wrong unit
returns a plausible number rather than an error; `find_nk_index` raises a warning when this
happens, but the warning is the only signal.

## refractiveindex_info/

Converted from the [refractiveindex.info database](https://github.com/polyanskiy/refractiveindex.info-database)
(CC0). Each file's header comments carry the original reference, the measurement conditions, the
upstream path and the conversion date.

A material name alone does not identify a dataset -- these records differ in sample preparation,
temperature and coverage. Where more than one is shipped, pick deliberately.

| key | range (m) | reference | sample |
| --- | --- | --- | --- |
| `si_aspnes` | 2.066e-07 – 8.266e-07 | Aspnes & Studna 1983 | c-Si <111>, room temperature |
| `si_green-2008` | 2.500e-07 – 1.450e-06 | Green 2008 | intrinsic Si, 300 K, KK-consistent |
| `sio2_malitson` | 2.100e-07 – 6.700e-06 | Malitson 1965 | fused silica, 20 °C |
| `sio2_lemarchand` | 2.500e-07 – 2.500e-06 | Lemarchand 2013 | 580 nm sputtered film on BK7 |
| `si3n4_luke` | 3.100e-07 – 5.504e-06 | Luke et al. 2015 | 340 nm Si₃N₄ on thermal SiO₂ |
| `al2o3_malitson-o` / `-e` | 2.000e-07 – 5.000e-06 | Malitson 1962 | synthetic sapphire |
| `al2o3_querry-o` / `-e` | 2.100e-07 – 5.556e-05 | Querry 1985 | sapphire, reaches 55 µm and has k |
| `sio2_ghosh-o` / `-e` | 1.980e-07 – 2.053e-06 | Ghosh 1999 | α-quartz |
| `tio2_bond-o` / `-e` | 4.500e-07 – 2.400e-06 | Bond 1965 | rutile |
| `batio3_wemple-o` / `-e` | 4.000e-07 – 7.000e-07 | Wemple 1968 | BaTiO₃ |
| `tio2_siefke` | 1.202e-07 – 1.251e-04 | Siefke et al. 2016 | 350 nm ALD film |
| `hfo2_franta` | 1.148e-07 – 1.251e-04 | Franta et al. 2015 | e-beam evaporated film on c-Si |
| `au_johnson` | 1.879e-07 – 1.937e-06 | Johnson & Christy 1972 | room temperature |
| `au_mcpeak` | 3.000e-07 – 1.700e-06 | McPeak et al. 2015 | template-stripped evaporated film |
| `ag_johnson` | 1.879e-07 – 1.937e-06 | Johnson & Christy 1972 | room temperature |
| `al_mcpeak` | 1.500e-07 – 1.700e-06 | McPeak et al. 2015 | template-stripped evaporated film |
| `cu_johnson` | 1.879e-07 – 1.937e-06 | Johnson & Christy 1972 | room temperature |
| `ti_johnson` | 1.880e-07 – 1.937e-06 | Johnson & Christy 1974 | room temperature |
| `ti_rakic-bb` | 2.480e-07 – 3.100e-05 | Rakić et al. 1998 | Brendel-Bormann model |
| `gan_kawashima` | 1.240e-07 – 9.919e-07 | Kawashima et al. 1997 | hexagonal GaN on sapphire |
| `mgo_synowicki` | 1.301e-07 – 3.300e-05 | Synowicki & Tiwald 2004 | oscillator model |
| `batio3_johnston-clamped` | 4.000e-07 – 1.000e-06 | Johnston 1971 | crystalline BaTiO₃, clamped |
| `ito_minenkov-glass` | 1.915e-07 – 1.689e-06 | Minenkov et al. 2024 | 110 nm ITO on glass |
| `graphene_el-sayed` | 2.400e-07 – 1.000e-06 | El-Sayed et al. 2021 | CVD graphene |
| `graphene_song` | 1.930e-07 – 1.690e-06 | Song et al. 2018 | CVD mono-graphene |

### Measurements and fits are stored differently

Optical constants come in two shapes and are kept that way. Measurements are tabulated, so those
files carry data rows. Fits are a handful of Sellmeier coefficients standing in for a curve, so
those files carry the coefficients and no rows at all:

```
Wavelength(m)	n	k
# material: SiO2
# type: formula 1
# coefficients: 0 0.6961663 6.84043e-08 0.4079426 1.162414e-07 0.8974794 9.896161e-06
# coefficient_unit: m
# range: 2.1000e-07 - 6.7000e-06 m
```

`read_material_table` reads the header, and `find_nk_index` evaluates the formula instead of
interpolating. Both kinds are used the same way from outside. Expanding a fit onto a grid would
lose the exact curve, cap it at whatever range was sampled, and turn seven coefficients into five
thousand rows, so it is not done. A Sellmeier fit gives `n` alone; `k` is 0 over the range it
covers.

Formula numbering follows refractiveindex.info. `meent/dispersion.py` implements formulas 1
(Sellmeier) and 2 (Sellmeier-2); anything else raises `NotImplementedError` rather than guessing.

Upstream quotes the coefficients against micrometres; the converter rescales the ones that carry
a length so the fit takes a wavelength in **metres**, like everything else here, and the
`# coefficient_unit:` line records that. Only the pole terms have a dimension -- a length in
formula 1, a length squared in formula 2 -- so those are the only ones that differ from the
published numbers. The oscillator strengths are dimensionless and unchanged.

The oscillator models the lab's MATLAB routines compute inline are stored the same way, under a
`type: model <name>` header instead of a formula number:

```
Wavelength(m)	n	k
# material: Au
# source: Material_data set, optprop_Au.m
# type: model brendel-bormann-ev
# coefficients: 9.20007 0.85195 0.09509 ...
# range: 1.5000e-07 - 1.0000e-04 m
```

`dispersion.py` implements `brendel-bormann-ev` (parameters in eV, as `optprop_Au.m` and
`optprop_Ti.m` quote them) and `brendel-bormann-cm` (in wavenumbers, as `optprop_CaF2.m` does).
These are the one place metres do not reach: an oscillator's damping or Gaussian width is a
frequency, not a length, so there is no metre form to convert it to. The lookup wavelength is
still metres -- the model converts it to the frequency it needs.
Unlike the Sellmeier fits these return a complex `n + ik` -- absorption is the point of them --
and `find_nk_index` flips the sign to the `n - ik` meent solves on, exactly as it does for a
tabulated `k` column. They need the Faddeeva function from scipy, so they evaluate through numpy:
fine on every backend, but a jax tracer cannot pass through one.

A file is only read as a formula if it actually carries a `coefficients` line. A header that
merely documents the model a table was sampled from stays a table.

### Birefringent materials

A table holds one refractive index, so uniaxial crystals ship as separate ordinary (`-o`) and
extraordinary (`-e`) files, the way upstream stores them. Putting both in one file does not work:
`find_nk_index` reads columns 1 and 2, so extra columns are dropped without a warning and you get
the ordinary ray while believing otherwise.

Combine them yourself, since `put_refractive_index_in_ucell` returns an array shaped like the
pattern it was given and cannot produce the extra axis:

```python
n_o = meent.find_nk_index('tio2_bond-o', table, wl)
n_e = meent.find_nk_index('tio2_bond-e', table, wl)

ucell = torch.zeros((1, 1, 10, 3), dtype=torch.complex128)   # last axis = (nx, ny, nz)
ucell[..., 0] = ucell[..., 1] = n_o                          # optic axis along z
ucell[..., 2] = n_e
```

The solver reads that last axis as the diagonal of the permittivity tensor. An off-diagonal
tensor -- an optic axis that is not aligned with x, y or z -- is not supported.

Two caveats on the data:

`al2o3_querry-*` quotes n to three decimals, which cannot resolve sapphire's 0.008 birefringence;
subtracting the two rays gives a rounding artefact of about 0.024 that barely varies with
wavelength. Use it for a single ray, where its reach to 55 µm and its k are what it is good for,
and use `al2o3_malitson-*` for birefringence.

GaN is shipped only as `gan_kawashima`, a single index. Its one upstream ordinary/extraordinary
pair fits the two rays with different functional forms -- the extraordinary one has no
visible-range oscillator at all -- and their difference comes out negative where wurtzite GaN is
positive uniaxial, so it is left out rather than shipped as a birefringent pair.

## jLab/

Converted from the lab's MATLAB `optprop_*` routines.

**Each file is named after the source it came from**, extension changed to `.txt`, and that name
is the lookup key. Where one source yields more than one file -- a birefringent pair, a model
family -- the only thing appended is what distinguishes them.

| key | range (m) | origin |
| --- | --- | --- |
| `wavnk_Si_JwKang_260409` | 1.251e-06 – 3.988e-05 | in-house measurement, 2026-04-09 |
| `wavnk_SiO2_JwKang_260409` | 1.252e-06 – 3.988e-05 | in-house measurement, 2026-04-09 |
| `wavnk_MgO_Palik` | 1.319e-06 – 1.370e-05 | Palik handbook |
| `260223_BTO_Vis_MIR_intrinsic-o` / `-e` | 1.930e-07 – 3.988e-05 | in-house measurement, 2026-02-23 |
| `[Optprop][Iongel][JW]` | 1.667e-06 – 3.324e-05 | ion gel, measured |
| `optprop_Au` | 1.500e-07 – 1.000e-04 | Brendel-Bormann model, **coefficients** |
| `optprop_Ti` | 1.500e-07 – 1.000e-04 | Brendel-Bormann model, **coefficients** |
| `optprop_CaF2` | 1.500e-07 – 1.000e-04 | Brendel-Bormann model, **coefficients** |
| `optprop_Gr_Falkovsky-ef200meV` | 5.000e-07 – 5.000e-05 | Falkovsky/Kubo model, sampled |
| `optprop_Gr_Falkovsky-ef400meV` | 5.000e-07 – 5.000e-05 | Falkovsky/Kubo model, sampled |
| `optprop_Gr_Falkovsky-ef600meV` | 5.000e-07 – 5.000e-05 | Falkovsky/Kubo model, sampled |

Several of these are models rather than measurements, so they carry assumptions a table cannot
show:

`optprop_Gr_Falkovsky-*` is one curve out of a model family. Graphene's response is set by how it
is gated, so each file fixes a Fermi level (the interband onset sits at `hw = 2 Ef`) alongside a
mobility of 10000 cm²/Vs and 300 K. The model gives a sheet conductivity; the tables spread it
over an assumed 0.34 nm thickness to get a bulk index, so a layer using them has to be that
thick to carry the intended sheet response. For other gating, edit `GRAPHENE_FERMI_LEVELS` in
the converter and re-run. The measured `graphene_el-sayed` and `graphene_song` tables are the
better choice when the graphene is not being tuned.

For arbitrary gating, mobility and temperature, build a dynamic material instead of selecting a
sampled table:

```python
graphene = meent.graphene_falkovsky(
    fermi_level=0.35,  # eV
    mobility=8000,     # cm^2/(V s)
    temperature=325,   # K
)

nk = meent.find_nk_index(graphene, meent.read_material_table(), 1.55e-6)
mee.ucell = mee.put_refractive_index_in_ucell(ucell, [1.0, graphene], 1.55e-6)
```

The optional `thickness` argument defaults to 0.34 nm, matching the MATLAB source.  This is an
effective bulk index obtained from sheet conductivity, so use the same physical layer thickness
in the geometry.  The dynamic model uses SciPy quadrature and accepts concrete scalar conditions;
it is suitable for arbitrary/random sampling but is not differentiable through PyTorch or JAX.

`260223_BTO_Vis_MIR_intrinsic-*` is uniaxial and ships as an ordinary/extraordinary pair, like the
crystals under `refractiveindex_info/`. It spans the visible into the mid-infrared and its
extinction is real above roughly 15 µm, where BaTiO₃'s phonon resonances take over -- at 20 µm the
two rays sit at 1.17 - 3.21i and 2.88 - 1.79i, so which axis the field sees changes the answer
completely there.

`optprop_Au`, `optprop_Ti` and `optprop_CaF2` compute a Brendel-Bormann oscillator model inline --
a Drude term plus Lorentz oscillators broadened by a Gaussian spread in centre frequency, which
integrates to a Voigt profile and so needs the Faddeeva function. Like the Sellmeier fits, they
are stored as coefficients and evaluated on lookup, so the file holds no rows to fall back on and
the parameters stay editable in place. The one cost is that `wofz` comes from scipy, so these
evaluate through numpy rather than through a backend's own array library.

`optprop_Ti` reproduces the refractiveindex.info `ti_rakic-bb` record to about 1e-3, so it is
another copy of the same model rather than new data. **`optprop_Au` does not match any measured
gold in this repository**: its n runs 0.5 to 0.8 high against Johnson & Christy, McPeak and Olmon
alike, and its coefficients are not the published Rakić set either. What it was fit to is not
recorded. Prefer `au_johnson` or `au_mcpeak` unless you know this fit is the right one for your
sample.

## Adding a material

These tables were generated from upstream sources by conversion scripts that are not shipped with
meent. A new material is added by writing the file directly, in one of two forms.

Both start with the same tab-separated column line and a block of `# key: value` header comments.
`# range:` is the wavelength span the entry is valid over, in metres, and `find_nk_index` warns
when a lookup falls outside it.

**Tabulated** -- `# type: tabulated nk` (or `tabulated n` / `tabulated k`), followed by data rows
of wavelength, n, k:

```
Wavelength(m)	n	k
# material: Ag
# source: P. B. Johnson and R. W. Christy. ... Phys. Rev. B 6, 4370-4379 (1972)
# type: tabulated nk
# range: 1.8790e-07 - 1.9370e-06 m
1.879000e-07	1.07	1.212
1.916000e-07	1.1	1.232
```

**Dispersion fit** -- `# type: formula <n>` plus a `# coefficients:` line and no data rows.
`meent/dispersion.py` evaluates it at lookup time, so the entry stays exact at any wavelength in
range instead of being interpolated:

```
Wavelength(m)	n	k
# material: SiO2
# type: formula 1
# coefficients: 0.0 0.6961663 6.84043e-08 0.4079426 1.16241e-07 0.8974794 9.89616e-06
# coefficient_unit: m
# range: 2.1000e-07 - 6.7000e-06 m
```

Supported forms are Sellmeier-type formulas 1 and 2, Brendel-Bormann (`brendel-bormann-ev` and
`brendel-bormann-cm`), and `graphene-falkovsky`. Adding another means implementing it in
`meent/dispersion.py`; unimplemented ones raise `NotImplementedError` on lookup rather than being
guessed at.

Absorbing materials are stored with a **positive** `k` column -- the ordinary optics `n + ik`
convention, the same one refractiveindex.info and the lab's MATLAB tables publish. `find_nk_index`
applies the sign on read, exactly as it does for the dispersion formulas above, and returns the
`n - ik` meent solves on.

So a table copied from refractiveindex.info is used as it comes: do not flip `k`. A table that
already carries a negative `k` is the one to correct, or it will produce gain instead of loss
with no warning.

New folders also have to be added to `package_data` in `setup.py`, or they will be missing from
an installed copy.

Tables must stay pure ASCII: `numpy.loadtxt` opens files with the locale default encoding, so a
non-ASCII byte breaks loading on machines whose locale is not UTF-8.

Keys are file names, and `read_material_table` walks every folder into one dict, so a name used
twice silently shadows. Under `refractiveindex_info/` that means `<material>_<source>.txt`; under
`jLab/` the source file name already serves, since it carries the material and the dataset both.

## Citation

If you publish results using the `refractiveindex_info/` tables, cite the database:

> Polyanskiy, M. N. *Refractiveindex.info database of optical constants*.
> *Scientific Data* **11**, 94 (2024). https://doi.org/10.1038/s41597-023-02898-2

and, where appropriate, the original measurement listed in each file's `# source:` line.
