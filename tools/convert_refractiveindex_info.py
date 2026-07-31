"""Convert refractiveindex.info YAML records into meent nk_data text tables.

The refractiveindex.info database (CC0, https://github.com/polyanskiy/refractiveindex.info-database)
stores optical constants as YAML, either as tabulated (wavelength, n, k) rows or as a
dispersion formula. meent reads plain text tables instead, so this script flattens the
selected records into the format read by ``read_material_table``:

    Wavelength(m)<TAB>n<TAB>k
    # provenance comments
    <rows>

Wavelengths are written in metres. The upstream files use micrometres.

Usage:
    python tools/convert_refractiveindex_info.py <path to database/data> [-o <out dir>]

The database snapshot is not vendored; download it once with

    curl -L -o ridb.tar.gz \
        https://github.com/polyanskiy/refractiveindex.info-database/archive/refs/heads/main.tar.gz

and point this script at ``<extracted>/database/data``.
"""

import argparse
import re
import unicodedata
from datetime import date
from pathlib import Path

import numpy as np
import yaml

UPSTREAM = 'polyanskiy/refractiveindex.info-database@ff11b58'

# Number of samples used to flatten a dispersion formula into a table. Linear interpolation
# error falls off as the square of the spacing; 5000 points keeps it under the 6 significant
# figures the tables are written with, even across the steep UV end of a Sellmeier curve.
FORMULA_SAMPLES = 5000

# (output name, path under database/data, record) -- one entry per shipped table.
# Records are chosen for being the standard reference for that material and for covering the
# UV-NIR range meent is normally used in. Where a material has several widely used datasets,
# more than one is shipped so the caller can pick.
RECORDS = [
    ('Si_aspnes', 'main/Si', 'Aspnes'),
    ('Si_green-2008', 'main/Si', 'Green-2008'),
    ('SiO2_malitson', 'main/SiO2', 'Malitson'),
    ('SiO2_lemarchand', 'main/SiO2', 'Lemarchand'),
    ('Si3N4_luke', 'main/Si3N4', 'Luke'),
    ('Al2O3_malitson', 'main/Al2O3', 'Malitson'),
    ('TiO2_siefke', 'main/TiO2', 'Siefke'),
    ('HfO2_franta', 'main/HfO2', 'Franta'),
    ('Au_johnson', 'main/Au', 'Johnson'),
    ('Au_mcpeak', 'main/Au', 'McPeak'),
    ('Ag_johnson', 'main/Ag', 'Johnson'),
    ('Al_mcpeak', 'main/Al', 'McPeak'),
    ('Cu_johnson', 'main/Cu', 'Johnson'),
    ('Ti_johnson', 'main/Ti', 'Johnson'),
    ('GaN_kawashima', 'main/GaN', 'Kawashima'),
    ('MgO_synowicki', 'main/MgO', 'Synowicki'),
    ('BaTiO3_johnston-clamped', 'main/BaTiO3', 'Johnston-clamped'),
    ('ITO_minenkov-glass', 'other/mixed crystals/In2O3-SnO2', 'Minenkov-glass'),
    ('Graphene_el-sayed', 'main/C', 'El-Sayed'),
    ('Graphene_song', 'main/C', 'Song'),
]


# numpy.loadtxt opens files with the locale default encoding, so the tables have to stay pure
# ASCII to load on any machine. Upstream references carry accented author names and unicode
# symbols; fold them rather than dropping the citation.
_ASCII_REPLACEMENTS = {
    'µ': 'u', 'μ': 'u',                                    # micro sign, greek mu
    '–': '-', '—': '-', '−': '-',                     # dashes
    '×': 'x', '±': '+/-', '°': 'deg', 'Å': 'A',  # symbols
    '‘': "'", '’': "'", '“': '"', '”': '"',      # quotes
}


def _flatten(text):
    """Collapse a YAML block string with HTML markup into one plain ASCII line."""
    if not text:
        return ''
    text = re.sub(r'<[^>]+>', '', text)
    for source, target in _ASCII_REPLACEMENTS.items():
        text = text.replace(source, target)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    return ' '.join(text.split())


def _formula1(coefficients, wl_um):
    """Sellmeier: n^2 - 1 = C0 + sum_i C_2i-1 * L^2 / (L^2 - C_2i^2), L in micrometres."""
    wl2 = np.asarray(wl_um, dtype=float) ** 2
    n2 = 1.0 + coefficients[0]
    for i in range(1, len(coefficients), 2):
        n2 = n2 + coefficients[i] * wl2 / (wl2 - coefficients[i + 1] ** 2)
    return np.sqrt(n2)


def _rows(block):
    return np.array([[float(v) for v in line.split()] for line in block.strip().splitlines()])


def _read_record(path):
    """Return (wavelength_um, n, k, kind) for one YAML record.

    n and k are sampled on a common wavelength grid. Records that only specify n (dispersion
    formulas and 'tabulated n' entries) get k = 0, which is what the source implies: the data
    was taken in a range where the material is treated as transparent.
    """
    document = yaml.safe_load(path.read_text(encoding='utf-8'))

    wl = n = k = None
    kinds = []
    for entry in document['DATA']:
        kind = entry['type']
        kinds.append(kind)

        if kind == 'tabulated nk':
            table = _rows(entry['data'])
            wl, n, k = table[:, 0], table[:, 1], table[:, 2]
        elif kind == 'tabulated n':
            table = _rows(entry['data'])
            wl, n = table[:, 0], table[:, 1]
        elif kind == 'tabulated k':
            table = _rows(entry['data'])
            # k is often tabulated on its own grid; move it onto the n grid.
            k = np.interp(wl, table[:, 0], table[:, 1]) if wl is not None else table[:, 1]
            if wl is None:
                wl = table[:, 0]
        elif kind == 'formula 1':
            lo, hi = (float(v) for v in str(entry['wavelength_range']).split())
            coefficients = [float(v) for v in str(entry['coefficients']).split()]
            wl = np.linspace(lo, hi, FORMULA_SAMPLES)
            n = _formula1(coefficients, wl)
        else:
            raise NotImplementedError(f'{path.name}: unsupported record type {kind!r}')

    if wl is None or n is None:
        raise ValueError(f'{path.name}: no refractive index data')
    if k is None:
        k = np.zeros_like(n)

    return wl, n, k, ', '.join(kinds)


def convert(name, shelf, page, data_root, out_dir):
    source = Path(data_root) / shelf / 'nk' / f'{page}.yml'
    document = yaml.safe_load(source.read_text(encoding='utf-8'))
    wl_um, n, k, kind = _read_record(source)

    header = [f'# material: {name.split("_")[0]}']
    header.append(f'# source: {_flatten(document.get("REFERENCES"))}')
    if document.get('COMMENTS'):
        header.append(f'# conditions: {_flatten(document["COMMENTS"])}')
    header.append(f'# upstream: {UPSTREAM} {shelf}/nk/{page}.yml')
    header.append(f'# type: {kind}')
    if 'formula' in kind:
        header.append(f'# note: dispersion formula sampled at {FORMULA_SAMPLES} points; k set to 0')
    elif 'tabulated nk' not in kind:
        header.append('# note: k not given by the source; set to 0')
    header.append(f'# range: {wl_um[0] * 1e-6:.4e} - {wl_um[-1] * 1e-6:.4e} m')
    header.append(f'# converted: {date.today().isoformat()} by tools/convert_refractiveindex_info.py'
                  ' (wavelength um -> m)')

    lines = ['Wavelength(m)\tn\tk']
    lines += header
    for wl_value, n_value, k_value in zip(wl_um, n, k):
        lines.append(f'{wl_value * 1e-6:.6e}\t{n_value:.6g}\t{k_value:.6g}')

    out_path = Path(out_dir) / f'{name}.txt'
    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return out_path, len(wl_um), wl_um[0] * 1e-6, wl_um[-1] * 1e-6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_root', help='path to <snapshot>/database/data')
    parser.add_argument('-o', '--out-dir',
                        default=str(Path(__file__).resolve().parent.parent
                                    / 'meent' / 'nk_data' / 'refractiveindex_info'))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, shelf, page in RECORDS:
        path, count, lo, hi = convert(name, shelf, page, args.data_root, out_dir)
        print(f'{path.name:28} {count:5} pts  {lo:.3e} - {hi:.3e} m')


if __name__ == '__main__':
    main()
