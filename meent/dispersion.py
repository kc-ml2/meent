"""Dispersion formulas used by the nk_data tables.

Optical constants come in two shapes. Most are measurements, tabulated against wavelength, and
those are stored as data rows. The rest are fits -- a handful of Sellmeier coefficients standing
in for a curve -- and expanding those onto a grid would throw away the exact form to gain
nothing, so they are stored as coefficients and evaluated on lookup instead.

Formula numbering follows refractiveindex.info, whose files these came from. Coefficients keep
that database's convention of wavelength in micrometres, while the tables around them are in
metres; `evaluate` converts, so callers work in metres throughout.

The maths is plain arithmetic and a square root, so it runs under numpy or jax alike -- pass the
module as `xp` to keep jax tracing intact.
"""


def formula_1(coefficients, wl_um, xp):
    """Sellmeier: n^2 - 1 = C0 + sum_i C_2i-1 * L^2 / (L^2 - C_2i^2)."""
    wl2 = xp.asarray(wl_um) ** 2
    n2 = 1.0 + coefficients[0]
    for i in range(1, len(coefficients), 2):
        n2 = n2 + coefficients[i] * wl2 / (wl2 - coefficients[i + 1] ** 2)
    return xp.sqrt(n2)


def formula_2(coefficients, wl_um, xp):
    """Sellmeier-2: as formula 1, but the pole term is C_2i rather than C_2i squared."""
    wl2 = xp.asarray(wl_um) ** 2
    n2 = 1.0 + coefficients[0]
    for i in range(1, len(coefficients), 2):
        n2 = n2 + coefficients[i] * wl2 / (wl2 - coefficients[i + 1])
    return xp.sqrt(n2)


FORMULAS = {1: formula_1, 2: formula_2}


def evaluate(spec, wl, xp):
    """Refractive index from a formula spec at wavelength wl, in metres.

    A dispersion formula gives n alone; k is zero, which is what the fit implies -- these are
    ranges where the material is treated as transparent.
    """
    formula = FORMULAS.get(spec['formula'])
    if formula is None:
        raise NotImplementedError(f"dispersion formula {spec['formula']} is not implemented")
    return formula(spec['coefficients'], xp.asarray(wl) * 1e6, xp)


def parse_header(path):
    """Read a table's '# key: value' comments, returning a formula spec if it carries one.

    Returns None for ordinary tables, so the caller can fall back to loading data rows.
    """
    entry = {}
    with open(path) as file:
        next(file, None)  # column header
        for line in file:
            if not line.startswith('#'):
                break
            key, _, value = line[1:].partition(':')
            entry[key.strip()] = value.strip()

    kind = entry.get('type', '')
    if not kind.startswith('formula'):
        return None

    # '2.1000e-07 - 6.7000e-06 m': split on the separating dash, not the one in an exponent.
    low, high = entry['range'].split(' - ')
    return {
        'formula': int(kind.split()[1]),
        'coefficients': [float(v) for v in entry['coefficients'].split()],
        'wavelength_range': (float(low), float(high.split()[0])),
    }
