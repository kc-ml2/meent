# nk_data

Refractive index tables read by `read_material_table()`. Any `.txt` or `.mat` file below this
directory is picked up; the **file name is the lookup key**, case-insensitive, and the directory
it sits in is ignored.

```python
import meent

mee = meent.call_mee(backend=0, wavelength=1.55e-6, period=[1.0e-6], thickness=[2.0e-7])
mee.ucell = mee.put_refractive_index_in_ucell(ucell, ['sio2_malitson', 'au_johnson'], 1.55e-6)
```

Use `meent.print_materials()` to list everything available with its wavelength range, or
`meent.list_materials()` for the same data as dicts.

## Wavelength units differ between folders

meent itself is unit-agnostic -- only the ratio of wavelength to period matters -- so these
tables are the one place an absolute unit is fixed.

| folder | unit |
| --- | --- |
| `refractiveindex_info/` | **metres** |
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
| `al2o3_malitson` | 2.652e-07 – 5.577e-06 | Malitson 1962 | synthetic sapphire, ordinary ray |
| `tio2_siefke` | 1.202e-07 – 1.251e-04 | Siefke et al. 2016 | 350 nm ALD film |
| `hfo2_franta` | 1.148e-07 – 1.251e-04 | Franta et al. 2015 | e-beam evaporated film on c-Si |
| `au_johnson` | 1.879e-07 – 1.937e-06 | Johnson & Christy 1972 | room temperature |
| `au_mcpeak` | 3.000e-07 – 1.700e-06 | McPeak et al. 2015 | template-stripped evaporated film |
| `ag_johnson` | 1.879e-07 – 1.937e-06 | Johnson & Christy 1972 | room temperature |
| `al_mcpeak` | 1.500e-07 – 1.700e-06 | McPeak et al. 2015 | template-stripped evaporated film |
| `cu_johnson` | 1.879e-07 – 1.937e-06 | Johnson & Christy 1972 | room temperature |
| `ti_johnson` | 1.880e-07 – 1.937e-06 | Johnson & Christy 1974 | room temperature |
| `gan_kawashima` | 1.240e-07 – 9.919e-07 | Kawashima et al. 1997 | hexagonal GaN on sapphire |
| `mgo_synowicki` | 1.301e-07 – 3.300e-05 | Synowicki & Tiwald 2004 | oscillator model |
| `batio3_johnston-clamped` | 4.000e-07 – 1.000e-06 | Johnston 1971 | crystalline BaTiO₃, clamped |
| `ito_minenkov-glass` | 1.915e-07 – 1.689e-06 | Minenkov et al. 2024 | 110 nm ITO on glass |
| `graphene_el-sayed` | 2.400e-07 – 1.000e-06 | El-Sayed et al. 2021 | CVD graphene |
| `graphene_song` | 1.930e-07 – 1.690e-06 | Song et al. 2018 | CVD mono-graphene |

`sio2_malitson`, `si3n4_luke` and `al2o3_malitson` are stored upstream as Sellmeier
coefficients rather than measurements; they are sampled onto a table at conversion time and
carry `k = 0`, which is what a dispersion formula implies. Their `# note:` header says so.

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
handles `tabulated nk`, `tabulated n`, `tabulated k` and `formula 1` (Sellmeier). Other
dispersion formulas raise `NotImplementedError` rather than guessing.

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
