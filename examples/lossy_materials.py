"""Example: Using lossy (complex refractive index) materials in meent.

Meent uses the exp(+iwt) time-harmonic convention internally.
Standard optical data (e.g., refractiveindex.info, Palik, Johnson & Christy)
uses exp(-iwt), where Im(n) > 0 indicates optical loss.

This means the sign of the imaginary part must be handled carefully
depending on WHERE the refractive index is used:

  +------------------+------------------+-------------------------------------+
  | Parameter        | What to pass     | Why                                 |
  +------------------+------------------+-------------------------------------+
  | n_top, n_bot     | raw n (as-is)    | Internal kz calculation already     |
  |                  |                  | applies conjugation.                |
  +------------------+------------------+-------------------------------------+
  | ucell (raster)   | conj(n)          | meent squares it: eps = n**2.       |
  |                  |                  | conj(n)**2 gives Im(eps) < 0,       |
  |                  |                  | which is loss in exp(+iwt).         |
  +------------------+------------------+-------------------------------------+
  | ucell (vector)   | conj(n)          | Same as raster — pass conjugated n. |
  +------------------+------------------+-------------------------------------+

WARNING: Passing conjugated n to n_top or n_bot causes double conjugation,
which can lead to unphysical results (e.g., R > 1).

This example simulates a 1D uniform Au film and compares with
the analytical Fresnel equation to verify correctness.
"""

import numpy as np
import meent


def run():
    wavelength = 1500  # nm
    # Au refractive index at 1500nm (Johnson & Christy, exp(-iwt) convention)
    n_au = 0.4941 + 10.3528j  # Im > 0 = lossy
    n_air = 1.0

    print(f"Au @ {wavelength}nm: n = {n_au}")
    print(f"  Standard convention: Im(n) > 0 means optical loss")

    # --- Analytical Fresnel (for reference) ---
    r_fresnel = (n_air - n_au) / (n_air + n_au)
    R_fresnel = abs(r_fresnel) ** 2
    print(f"\nFresnel (air -> bulk Au): R = {R_fresnel:.6f}")

    # ------------------------------------------------------------------
    # Correct: uniform Au film as 1D raster
    # ------------------------------------------------------------------
    # ucell gets conj(n), n_bot gets raw n
    n_au_conj = np.conj(n_au)
    ucell = np.array([[[n_au_conj]]])  # uniform layer, single pixel

    mee = meent.call_mee(
        backend=0,
        pol=1,  # TM
        n_top=n_air,   # raw n
        n_bot=n_au,    # raw n — do NOT conjugate
        theta=0,
        fto=[0],       # no grating, fto=0 is sufficient
        wavelength=wavelength,
        period=[100],  # arbitrary (uniform film)
        ucell=ucell,
        thickness=[20],  # thin film — transmission is non-negligible
        type_complex=np.complex128,
    )
    result = mee.conv_solve()
    R = result.res.de_ri.sum()
    T = result.res.de_ti.sum()
    print(f"\nCorrect: ucell = conj(n_au), n_bot = n_au (raw)")
    print(f"  R = {R:.6f}, T = {T:.6f}, R+T = {R+T:.6f}")
    print(f"  R+T < 1: energy is absorbed by the lossy Au film")

    # ------------------------------------------------------------------
    # WRONG: conjugated n_bot
    # ------------------------------------------------------------------
    mee.n_bot = np.conj(n_au)  # WRONG
    result_wrong = mee.conv_solve()
    R_w = result_wrong.res.de_ri.sum()
    T_w = result_wrong.res.de_ti.sum()
    print(f"\nWRONG: n_bot = conj(n_au) — causes double conjugation")
    print(f"  R = {R_w:.6f}, T = {T_w:.6f}, R+T = {R_w+T_w:.6f}")
    if R_w + T_w > 1.01:
        print(f"  WARNING: R+T > 1 — energy conservation violated!")

    # ------------------------------------------------------------------
    # WRONG: raw n in ucell (not conjugated)
    # ------------------------------------------------------------------
    ucell_wrong = np.array([[[n_au]]])  # NOT conjugated
    mee.n_bot = n_au  # correct
    mee.ucell = ucell_wrong
    result_wrong2 = mee.conv_solve()
    R_w2 = result_wrong2.res.de_ri.sum()
    T_w2 = result_wrong2.res.de_ti.sum()
    print(f"\nWRONG: ucell = n_au (not conjugated)")
    print(f"  R = {R_w2:.6f}, T = {T_w2:.6f}, R+T = {R_w2+T_w2:.6f}")
    print(f"  R deviates from Fresnel: |dR| = {abs(R_w2 - R_fresnel):.6f}")


if __name__ == '__main__':
    run()
