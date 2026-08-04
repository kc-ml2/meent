"""Dispersion formulas used by the nk_data tables.

Optical constants come in two shapes. Most are measurements, tabulated against wavelength, and
those are stored as data rows. The rest are closed-form models -- a handful of coefficients
standing in for a curve -- and expanding those onto a grid would throw away the exact form to
gain nothing, so they are stored as coefficients and evaluated on lookup instead.

Two families live here:

  * `formula_*`, the Sellmeier fits, numbered as refractiveindex.info numbers them. These give a
    real n: they describe ranges where the material is treated as transparent.
  * `model_*`, the oscillator models the lab's MATLAB optprop_* routines compute inline. These
    give a complex index n + ik, because absorption is the point of them.

`evaluate` returns whichever the spec asks for, so callers must handle a complex result. The
sign convention is the ordinary optics one (n + ik); `find_nk_index` flips it to the n - ik that
meent solves on, exactly as it does for a tabulated k.

Everything with a length dimension is in metres: the wavelength passed in, and the Sellmeier
coefficients that are lengths. Upstream quotes those in micrometres, so the converter scales them
on the way in and the files record `coefficient_unit: m`.

The oscillator models are the one place metres do not reach. Their parameters are frequencies --
eV for optprop_Au/Ti, wavenumbers for optprop_CaF2 -- and an oscillator's damping or Gaussian
width is not a length, so it has no metre form to convert to. They stay as the sources quote
them, and the model turns the wavelength into the frequency it needs.

The Sellmeier maths is plain arithmetic and a square root, so it runs under numpy or jax alike --
pass the module as `xp` to keep jax tracing intact. The oscillator models additionally need the
Faddeeva function, which only scipy provides: they evaluate through numpy and so want a concrete
wavelength, not a jax tracer.
"""

import math

import numpy as np

WAVENUMBER_PER_EV = 8065.54429  # cm-1 per eV, the value the MATLAB sources hard-code

# Constants used by the Falkovsky/Kubo graphene model.  These deliberately match the values in
# the lab's optprop_Gr_Falkovsky.m so a dynamic lookup reproduces the existing sampled tables.
ELEMENTARY_CHARGE = 1.6021773349e-19
BOLTZMANN = 1.38064852e-23
HBAR = 1.0545718e-34
VACUUM_PERMITTIVITY = 8.85418781762039e-12
FERMI_VELOCITY = 1e6
LIGHT_SPEED = 299792458
GRAPHENE_THICKNESS = 0.34e-9


def graphene_falkovsky(fermi_level, mobility=10000.0, temperature=300.0,
                       thickness=GRAPHENE_THICKNESS):
    """Build a dynamic Falkovsky/Kubo graphene material specification.

    Args:
        fermi_level: Fermi level in eV.
        mobility: Carrier mobility in cm^2/(V s).
        temperature: Temperature in kelvin.
        thickness: Effective bulk thickness in metres.  The source model uses 0.34 nm.

    The returned object can be passed anywhere a material name is accepted, including
    ``find_nk_index`` and ``put_refractive_index_in_ucell``.  Conditions must be positive scalar
    values; wavelength is supplied later by the lookup or solver.
    """
    values = {
        'fermi_level': fermi_level,
        'mobility': mobility,
        'temperature': temperature,
        'thickness': thickness,
    }
    normalized = {}
    for name, value in values.items():
        try:
            value = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError(f'{name} must be a real scalar') from error
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be a finite positive value')
        normalized[name] = value

    return {
        'formula': 'graphene-falkovsky',
        'coefficients': [normalized['fermi_level'], normalized['mobility'],
                         normalized['temperature'], normalized['thickness']],
        # The source routine declares no fitted wavelength interval.  Positivity is checked by
        # the evaluator, so this suppresses the table-specific endpoint warning.
        'wavelength_range': (0.0, math.inf),
    }


def formula_1(coefficients, wl, xp):
    """Sellmeier: n^2 - 1 = C0 + sum_i C_2i-1 * L^2 / (L^2 - C_2i^2), with L and C_2i in metres."""
    wl2 = xp.asarray(wl) ** 2
    n2 = 1.0 + coefficients[0]
    for i in range(1, len(coefficients), 2):
        n2 = n2 + coefficients[i] * wl2 / (wl2 - coefficients[i + 1] ** 2)
    return xp.sqrt(n2)


def formula_2(coefficients, wl, xp):
    """Sellmeier-2: as formula 1, but the pole term is C_2i rather than C_2i squared, so C_2i is
    in metres squared."""
    wl2 = xp.asarray(wl) ** 2
    n2 = 1.0 + coefficients[0]
    for i in range(1, len(coefficients), 2):
        n2 = n2 + coefficients[i] * wl2 / (wl2 - coefficients[i + 1])
    return xp.sqrt(n2)


def _faddeeva(z):
    """w(z) = exp(-z^2) erfc(-iz). A Lorentz oscillator whose centre frequency is spread by a
    Gaussian integrates to a Voigt profile, which is where this comes from.

    Imported here rather than at module scope so that importing meent does not pull in
    scipy.special for the tables that never need it.
    """
    from scipy.special import wofz
    return wofz(z)


def model_brendel_bormann_ev(coefficients, wl, xp):
    """Brendel-Bormann as optprop_Au.m / optprop_Ti.m write it: a Drude term for the conduction
    electrons plus Gaussian-broadened Lorentz oscillators for the interband transitions.

    Coefficients are [wp, f0, g0, (f, gamma, centre, sigma) * n], all in eV.
    """
    omega = 1e-2 / xp.asarray(wl) / WAVENUMBER_PER_EV  # m -> cm-1 -> eV
    plasma, strength, damping = coefficients[:3]
    permittivity = 1 - strength * plasma ** 2 / (omega * (omega + 1j * damping))
    for i in range(3, len(coefficients), 4):
        f, gamma, centre, sigma = coefficients[i:i + 4]
        a = xp.sqrt(omega ** 2 + 1j * omega * gamma)
        permittivity = permittivity + 1j * math.sqrt(math.pi) * f * plasma ** 2 / (
            2 ** 1.5 * a * sigma) * (_faddeeva((a - centre) / (math.sqrt(2) * sigma))
                                     + _faddeeva((a + centre) / (math.sqrt(2) * sigma)))
    return xp.sqrt(permittivity)


def model_brendel_bormann_cm(coefficients, wl, xp):
    """The same oscillator sum as optprop_CaF2.m writes it: no Drude term, an explicit eps_inf,
    and parameters quoted in wavenumbers rather than eV.

    Coefficients are [eps_inf, (strength, gamma, sigma, centre) * n], all in cm-1.
    """
    omega = 1e-2 / xp.asarray(wl)  # m -> cm-1
    permittivity = coefficients[0]
    for i in range(1, len(coefficients), 4):
        strength, gamma, sigma, centre = coefficients[i:i + 4]
        a = xp.sqrt(omega ** 2 + 1j * gamma * omega)
        permittivity = permittivity + 1j * math.sqrt(math.pi) * strength / (
            2 * math.sqrt(2) * sigma * a) * (_faddeeva((a - centre) / (math.sqrt(2) * sigma))
                                             + _faddeeva((a + centre) / (math.sqrt(2) * sigma)))
    return xp.sqrt(permittivity)


def _cosh_ratio(a, b):
    """Stable cosh(a) / cosh(b) for the positive arguments in the Kubo occupation."""
    log_cosh_a = np.logaddexp(a, -a) - math.log(2)
    log_cosh_b = np.logaddexp(b, -b) - math.log(2)
    # A ratio above exp(709) is indistinguishable from infinity in the occupation denominator;
    # clipping avoids an overflow warning without changing the double-precision result.
    return np.exp(np.clip(log_cosh_a - log_cosh_b, -745, 709))


def model_graphene_falkovsky(coefficients, wl, xp):
    """Falkovsky/Kubo graphene index for arbitrary electrical and thermal conditions.

    Coefficients are ``[Fermi level (eV), mobility (cm^2/Vs), temperature (K),
    effective thickness (m)]``.  The quadrature follows optprop_Gr_Falkovsky.m.  It runs through
    NumPy/SciPy, so conditions and wavelengths must be concrete rather than autograd/JAX tracers.
    """
    from scipy.integrate import quad

    fermi_level, mobility, temperature, thickness = coefficients
    wavelength = np.asarray(wl, dtype=float)
    if np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0):
        raise ValueError('wavelength must contain only finite positive values in metres')

    shape = wavelength.shape
    wavelength = np.atleast_1d(wavelength).ravel()
    temperature_ev = temperature * BOLTZMANN / ELEMENTARY_CHARGE
    mobility_si = mobility * 1e-4
    photon_ev = HBAR * (2 * np.pi * LIGHT_SPEED / wavelength) / ELEMENTARY_CHARGE
    scattering = abs(HBAR * FERMI_VELOCITY ** 2
                     / (mobility_si * ELEMENTARY_CHARGE * fermi_level))

    def occupation(energy):
        return (np.tanh(energy / temperature_ev)
                / (_cosh_ratio(fermi_level / temperature_ev,
                               energy / temperature_ev) + 1.0))

    conductivity = np.empty(photon_ev.shape, dtype=complex)
    for index, energy in enumerate(photon_ev):
        half = occupation(energy / 2)
        integrand = lambda z: (occupation(z) - half) / (energy ** 2 - 4 * z * z)
        interband, _ = quad(integrand, 0, 1000, points=[energy / 2], limit=400)

        intraband = ((2j * ELEMENTARY_CHARGE ** 2 * temperature_ev)
                     / (np.pi * HBAR * (energy + 1j * scattering))
                     * np.logaddexp(fermi_level / (2 * temperature_ev),
                                    -fermi_level / (2 * temperature_ev)))
        conductivity[index] = intraband + (ELEMENTARY_CHARGE ** 2 / (4 * HBAR)) * (
            half + (4j * energy / np.pi) * interband)

    angular_frequency = ELEMENTARY_CHARGE * photon_ev / HBAR
    permittivity = 1 + 1j * conductivity / (
        angular_frequency * thickness * VACUUM_PERMITTIVITY)
    result = np.sqrt(permittivity).reshape(shape)
    if result.ndim == 0:
        return result.item()
    return result


FORMULAS = {
    1: formula_1,
    2: formula_2,
    'brendel-bormann-ev': model_brendel_bormann_ev,
    'brendel-bormann-cm': model_brendel_bormann_cm,
    'graphene-falkovsky': model_graphene_falkovsky,
}


def evaluate(spec, wl, xp):
    """Refractive index from a formula spec at wavelength wl, in metres.

    Sellmeier fits give a real n -- k is zero, which is what the fit implies. The oscillator
    models give a complex n + ik. Callers must handle both.
    """
    formula = FORMULAS.get(spec['formula'])
    if formula is None:
        raise NotImplementedError(f"dispersion formula {spec['formula']} is not implemented")
    return formula(spec['coefficients'], xp.asarray(wl), xp)


def parse_header(path):
    """Read a table's '# key: value' comments, returning a formula spec if it carries one.

    Two headers name a closed-form curve: 'type: formula N' for the Sellmeier fits and
    'type: model <name>' for the oscillator models. Either way the file must also carry a
    'coefficients' line -- a file that merely documents the model it was sampled from stays a
    table, which is what keeps the older sampled models loading as data rows.

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
    if 'coefficients' not in entry:
        return None
    if kind.startswith('formula'):
        name = int(kind.split()[1])
    elif kind.startswith('model'):
        name = kind.split(None, 1)[1].strip()
    else:
        return None

    # '2.1000e-07 - 6.7000e-06 m': split on the separating dash, not the one in an exponent.
    low, high = entry['range'].split(' - ')
    return {
        'formula': name,
        'coefficients': [float(v) for v in entry['coefficients'].split()],
        'wavelength_range': (float(low), float(high.split()[0])),
    }
