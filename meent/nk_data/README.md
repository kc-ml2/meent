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

## Wavelength units differ between folders

meent itself is unit-agnostic -- only the ratio of wavelength to period matters -- so these
tables are the one place an absolute unit is fixed.

| folder | unit |
| --- | --- |
| `refractiveindex_info/`, `jLab/` | **metres** |
| `filmetrics/`, `matlab/` | **nanometres** |

Do not mix the two in one `mat_list`. Interpolation clamps to the endpoints outside a table's
range, so a wavelength in the wrong unit returns a plausible number rather than an error;
`find_nk_index` raises a warning when this happens, but the warning is the only signal.

## refractiveindex_info/

Converted from the [refractiveindex.info database](https://github.com/polyanskiy/refractiveindex.info-database)
(CC0) by `tools/convert_refractiveindex_info.py`. Each file's header comments carry the original
reference, the measurement conditions, the upstream path and the conversion date.

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
# coefficients: 0 0.6961663 0.0684043 0.4079426 0.1162414 0.8974794 9.896161
# coefficient_unit: um
# range: 2.1000e-07 - 6.7000e-06 m
```

`read_material_table` reads the header, and `find_nk_index` evaluates the formula instead of
interpolating. Both kinds are used the same way from outside. Expanding a fit onto a grid would
lose the exact curve, cap it at whatever range was sampled, and turn seven coefficients into five
thousand rows, so it is not done. A fit gives `n` alone; `k` is 0 over the range it covers.

Formula numbering follows refractiveindex.info. `meent/dispersion.py` implements formulas 1
(Sellmeier) and 2 (Sellmeier-2); anything else raises `NotImplementedError` rather than guessing.
Coefficients keep upstream's micrometre convention while `range` is in metres, as the
`# coefficient_unit:` line records -- `evaluate` converts, so callers stay in metres.

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

Converted from the lab's MATLAB `optprop_*` routines by `tools/convert_matlab_optprop.py`.

| key | range (m) | origin |
| --- | --- | --- |
| `si_jwkang-260409` | 1.251e-06 – 3.988e-05 | in-house measurement, 2026-04-09 |
| `sio2_jwkang-260409` | 1.252e-06 – 3.988e-05 | in-house measurement, 2026-04-09 |
| `batio3_intrinsic-260223-o` / `-e` | 1.930e-07 – 3.988e-05 | in-house measurement, 2026-02-23 |
| `mgo_palik` | 1.319e-06 – 1.370e-05 | Palik handbook |
| `ti_brendelbormann` | 2.480e-07 – 3.100e-05 | same data as `ti_rakic-bb` |
| `iongel_jw` | 1.667e-06 – 3.324e-05 | ion gel, measured |
| `au_jw-bb` | 1.500e-07 – 1.000e-04 | Brendel-Bormann fit, sampled |
| `ti_jw-bb` | 1.500e-07 – 1.000e-04 | Brendel-Bormann fit, sampled |
| `caf2_jw-bb` | 1.500e-07 – 1.000e-04 | Brendel-Bormann fit, sampled |
| `au_palik-drude` | 5.000e-07 – 5.000e-05 | Drude fit, sampled |
| `graphene_falkovsky-ef200meV` | 5.000e-07 – 5.000e-05 | Falkovsky/Kubo model, sampled |
| `graphene_falkovsky-ef400meV` | 5.000e-07 – 5.000e-05 | Falkovsky/Kubo model, sampled |
| `graphene_falkovsky-ef600meV` | 5.000e-07 – 5.000e-05 | Falkovsky/Kubo model, sampled |

Two of these are models rather than measurements, so they carry assumptions a table cannot show:

`au_palik-drude` is a free-electron Drude fit with no interband term, despite the Palik in its
name. Gold's interband transitions dominate below roughly 500 nm, so this is an infrared model;
prefer `au_johnson` or `au_mcpeak` in the visible.

`graphene_falkovsky-*` is one curve out of a model family. Graphene's response is set by how it
is gated, so each file fixes a Fermi level (the interband onset sits at `hw = 2 Ef`) alongside a
mobility of 10000 cm²/Vs and 300 K. The model gives a sheet conductivity; the tables spread it
over an assumed 0.34 nm thickness to get a bulk index, so a layer using them has to be that
thick to carry the intended sheet response. For other gating, edit `GRAPHENE_FERMI_LEVELS` in
the converter and re-run. The measured `graphene_el-sayed` and `graphene_song` tables are the
better choice when the graphene is not being tuned.

`batio3_intrinsic-260223-*` is uniaxial and ships as an ordinary/extraordinary pair, like the
crystals under `refractiveindex_info/`. It spans the visible into the mid-infrared and its
extinction is real above roughly 15 µm, where BaTiO₃'s phonon resonances take over -- at 20 µm the
two rays sit at 1.17 - 3.21i and 2.88 - 1.79i, so which axis the field sees changes the answer
completely there.

`ti_brendelbormann` and `ti_rakic-bb` are the same numbers to the last digit: the MATLAB library's
copy came from the refractiveindex.info record. Both names are kept so the table is findable
either way, but there is no reason to prefer one over the other.

The `*_jw-bb` tables come from a newer `Material_data` set whose `optprop_*.m` compute a
Brendel-Bormann oscillator model inline -- a Drude term plus Lorentz oscillators broadened by a
Gaussian spread in centre frequency, which integrates to a Voigt profile and so needs the
Faddeeva function. They are sampled onto a grid rather than kept as coefficients, unlike the
Sellmeier fits under `refractiveindex_info/`: those evaluate with plain arithmetic in any array
library, while this one needs `wofz`, which jax does not provide.

`ti_jw-bb` reproduces `ti_brendelbormann` to about 1e-3, so it is a third copy of the same model
rather than new data. **`au_jw-bb` does not match any measured gold in this repository**: its n
runs 0.5 to 0.8 high against Johnson & Christy, McPeak and Olmon alike, and its coefficients are
not the published Rakić set either. What it was fit to is not recorded. Prefer `au_johnson` or
`au_mcpeak` unless you know this fit is the right one for your sample.

## Adding a material

Download a database snapshot once, add an entry to `RECORDS` in
`tools/convert_refractiveindex_info.py`, and re-run it:

```bash
curl -L -o ridb.tar.gz \
    https://github.com/polyanskiy/refractiveindex.info-database/archive/refs/heads/main.tar.gz
tar xzf ridb.tar.gz
python tools/convert_refractiveindex_info.py <extracted>/database/data
```

The converter needs `pyyaml`; meent itself does not, so it is not an install dependency. It
copies `tabulated nk`, `tabulated n` and `tabulated k` records out as data rows, and passes
dispersion formulas through as coefficients for `meent/dispersion.py` to evaluate. Adding a
formula means implementing it there; unimplemented ones raise `NotImplementedError` on lookup
rather than being guessed at.

New folders also have to be added to `package_data` in `setup.py`, or they will be missing from
an installed copy.

Tables must stay pure ASCII: `numpy.loadtxt` opens files with the locale default encoding, so a
non-ASCII byte breaks loading on machines whose locale is not UTF-8.

Name new files `<material>_<source>.txt`. A bare `<material>.txt` would collide with the
existing `filmetrics/` tables, and because keys are file names, one would silently shadow the
other.

## Citation

If you publish results using the `refractiveindex_info/` tables, cite the database:

> Polyanskiy, M. N. *Refractiveindex.info database of optical constants*.
> *Scientific Data* **11**, 94 (2024). https://doi.org/10.1038/s41597-023-02898-2

and, where appropriate, the original measurement listed in each file's `# source:` line.
