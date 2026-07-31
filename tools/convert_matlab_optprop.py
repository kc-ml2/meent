"""Convert the lab's MATLAB optprop_* material routines into meent nk_data text tables.

The MATLAB library under 'Material data' exposes one optprop_<material>_<source>.m per dataset.
Three shapes appear, and this script covers all of them:

  * thin wrappers that load 'material data library/wavnk_<name>.mat', whose 'data' variable is
    an (N, 3) array of [wavelength(um), n, k];
  * an inline Drude fit (optprop_Au_Palik);
  * a parametric Kubo model (optprop_Gr_Falkovsky), which is a model family rather than one
    material -- graphene's response depends on how it is gated, so a table has to fix the Fermi
    level, mobility and temperature.

Output goes to nk_data/jLab/. Tables whose upstream is the refractiveindex.info database
belong in nk_data/refractiveindex_info/ via tools/convert_refractiveindex_info.py instead --
Ti_BrendelBormann is one such case, being byte-identical to that database's Ti/Rakic-BB.

Usage:
    python tools/convert_matlab_optprop.py "<path to Material data>"
"""

import argparse
from datetime import date
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.io import loadmat

# Physical constants, matching the values hard-coded in the MATLAB sources so the ported models
# reproduce them exactly.
ELEMENTARY_CHARGE = 1.6021773349e-19
BOLTZMANN = 1.38064852e-23
HBAR = 1.0545718e-34
VACUUM_PERMITTIVITY = 8.85418781762039e-12
FERMI_VELOCITY = 1e6
LIGHT_SPEED = 299792458

# (output name, wavnk_*.mat stem) -- wrappers that just load a [wavelength(um), n, k] table.
MAT_TABLES = [
    ('MgO_palik', 'wavnk_MgO_Palik'),
    ('Si_jwkang-260409', 'wavnk_Si_JwKang_260409'),
    ('SiO2_jwkang-260409', 'wavnk_SiO2_JwKang_260409'),
    # Byte-identical to the refractiveindex.info Ti/Rakic-BB record, which ships as ti_rakic-bb.
    # Kept under the name the MATLAB library uses so it can be found either way.
    ('Ti_brendelbormann', 'wavnk_Ti_BrendelBormann'),
]

# Five-column tables, [wavelength(um), n_o, k_o, n_e, k_e], for uniaxial crystals. One table
# holds one index, so these are written out as separate <name>-o and <name>-e files: a single
# file would lose the extraordinary ray silently, since find_nk_index reads columns 1 and 2.
MAT_BIREFRINGENT = [
    ('BaTiO3_intrinsic-260223', 'wavnk_BTO_Intrinsic_260223'),
]

# Fermi levels tabulated for the Falkovsky model, in eV. Graphene is transparent below the
# interband onset at hw = 2 Ef, so each level gives a materially different curve.
GRAPHENE_FERMI_LEVELS = [0.2, 0.4, 0.6]
GRAPHENE_MOBILITY = 10000.0  # cm^2/Vs
GRAPHENE_TEMPERATURE = 300.0  # K
GRAPHENE_THICKNESS = 0.34e-9  # m, used to turn a sheet conductivity into a bulk permittivity

# Wavelength grid for the two models, log spaced because they span two decades.
MODEL_WAVELENGTH_RANGE = (0.5e-6, 50e-6)
MODEL_SAMPLES = 1000


def _cosh_ratio(a, b):
    """cosh(a) / cosh(b) for positive arguments, without overflowing at large b."""
    return np.exp(a - b) * (1 + np.exp(-2 * a)) / (1 + np.exp(-2 * b))


def graphene_falkovsky(wavelength, fermi_level, mobility, temperature):
    """Sheet conductivity of graphene from the Falkovsky/Kubo model, ported from MATLAB.

    Returns the complex refractive index obtained by spreading the sheet conductivity over
    GRAPHENE_THICKNESS, which is the convention the MATLAB routine uses.
    """
    temperature_ev = temperature * BOLTZMANN / ELEMENTARY_CHARGE
    mobility_si = mobility * 1e-4
    photon_ev = HBAR * (2 * np.pi * LIGHT_SPEED / wavelength) / ELEMENTARY_CHARGE

    # Impurity-limited scattering rate, in eV.
    scattering = abs(HBAR * FERMI_VELOCITY ** 2
                     / (mobility_si * ELEMENTARY_CHARGE * fermi_level))

    def occupation(energy):
        return (np.tanh(energy / temperature_ev)
                / (_cosh_ratio(fermi_level / temperature_ev, energy / temperature_ev) + 1.0))

    conductivity = np.empty(photon_ev.shape, dtype=complex)
    for index, energy in enumerate(photon_ev):
        half = occupation(energy / 2)

        # The pole at z = energy/2 is removable because the numerator vanishes there; quad still
        # needs to be told about it to place its panels sensibly.
        integrand = lambda z: (occupation(z) - half) / (energy ** 2 - 4 * z * z)
        interband, _ = quad(integrand, 0, 1000, points=[energy / 2], limit=400)

        intraband = ((2j * ELEMENTARY_CHARGE ** 2 * temperature_ev)
                     / (np.pi * HBAR * (energy + 1j * scattering))
                     * np.log(2 * np.cosh(fermi_level / (2 * temperature_ev))))
        conductivity[index] = intraband + (ELEMENTARY_CHARGE ** 2 / (4 * HBAR)) * (
            half + (4j * energy / np.pi) * interband)

    angular_frequency = ELEMENTARY_CHARGE * photon_ev / HBAR
    permittivity = 1 + 1j * conductivity / (
        angular_frequency * GRAPHENE_THICKNESS * VACUUM_PERMITTIVITY)
    return np.sqrt(permittivity)


def gold_drude(wavelength):
    """Drude fit used by optprop_Au_Palik: eps = eps_inf - wp^2 / (w^2 + i*Gamma*w)."""
    permittivity_infinity = 1.0
    plasma_frequency = 2 * np.pi * 1.85e15
    damping = 2 * np.pi * 14.5355e12

    angular_frequency = 2 * np.pi * LIGHT_SPEED / wavelength
    permittivity = permittivity_infinity - plasma_frequency ** 2 / (
        angular_frequency ** 2 + 1j * damping * angular_frequency)
    return np.sqrt(permittivity)


def write_table(out_dir, name, wavelength, n, k, header):
    lines = ['Wavelength(m)\tn\tk']
    lines += [f'# {line}' for line in header]
    lines.append(f'# range: {wavelength[0]:.4e} - {wavelength[-1]:.4e} m')
    lines.append(f'# converted: {date.today().isoformat()} by tools/convert_matlab_optprop.py')
    for wavelength_value, n_value, k_value in zip(wavelength, n, k):
        lines.append(f'{wavelength_value:.6e}\t{n_value:.6g}\t{k_value:.6g}')

    path = Path(out_dir) / f'{name}.txt'
    path.write_text('\n'.join(lines) + '\n', encoding='ascii')
    return path, len(wavelength)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('material_data', help="path to the MATLAB 'Material data' directory")
    parser.add_argument('-o', '--out-dir',
                        default=str(Path(__file__).resolve().parent.parent
                                    / 'meent' / 'nk_data' / 'jLab'))
    args = parser.parse_args()

    library = Path(args.material_data) / 'material data library'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, stem in MAT_TABLES:
        table = np.asarray(loadmat(str(library / f'{stem}.mat'))['data'], dtype=float)
        wavelength = table[:, 0] * 1e-6  # the MATLAB library tabulates micrometres
        path, count = write_table(
            out_dir, name, wavelength, table[:, 1], table[:, 2],
            [f'material: {name.split("_")[0]}',
             f'source: lab MATLAB library, {stem}.mat (via optprop_{stem[6:]}.m)',
             'type: tabulated nk'])
        print(f'{path.name:30} {count:5} pts')

    for name, stem in MAT_BIREFRINGENT:
        table = np.asarray(loadmat(str(library / f'{stem}.mat'))['data'], dtype=float)
        wavelength = table[:, 0] * 1e-6
        for ray, label, n_col, k_col in [('o', 'ordinary', 1, 2), ('e', 'extraordinary', 3, 4)]:
            path, count = write_table(
                out_dir, f'{name}-{ray}', wavelength, table[:, n_col], table[:, k_col],
                [f'material: {name.split("_")[0]}',
                 f'source: lab MATLAB library, {stem}.mat',
                 f'ray: {label}',
                 'note: uniaxial crystal; pair with the matching -o / -e table to build a '
                 '(nx, ny, nz) ucell',
                 'type: tabulated nk'])
            print(f'{path.name:30} {count:5} pts')

    wavelength = np.geomspace(*MODEL_WAVELENGTH_RANGE, MODEL_SAMPLES)

    index = gold_drude(wavelength)
    path, count = write_table(
        out_dir, 'Au_palik-drude', wavelength, index.real, index.imag,
        ['material: Au',
         'source: lab MATLAB library, optprop_Au_Palik.m (inline Drude fit)',
         'model: eps = 1 - wp^2/(w^2 + i*Gamma*w), wp = 2pi*1.85e15, Gamma = 2pi*14.5355e12',
         'note: free-electron Drude only, no interband term; unreliable below roughly 1 um',
         'type: model'])
    print(f'{path.name:30} {count:5} pts')

    for fermi_level in GRAPHENE_FERMI_LEVELS:
        index = graphene_falkovsky(wavelength, fermi_level,
                                   GRAPHENE_MOBILITY, GRAPHENE_TEMPERATURE)
        name = f'Graphene_falkovsky-ef{round(fermi_level * 1000)}meV'
        path, count = write_table(
            out_dir, name, wavelength, index.real, index.imag,
            ['material: Graphene',
             'source: lab MATLAB library, optprop_Gr_Falkovsky.m (Falkovsky/Kubo model)',
             f'conditions: Ef = {fermi_level} eV, mobility = {GRAPHENE_MOBILITY:g} cm^2/Vs, '
             f'T = {GRAPHENE_TEMPERATURE:g} K',
             f'note: sheet conductivity spread over {GRAPHENE_THICKNESS:g} m to give a bulk '
             'index; valid only at this gating',
             'type: model'])
        print(f'{path.name:30} {count:5} pts')


if __name__ == '__main__':
    main()
