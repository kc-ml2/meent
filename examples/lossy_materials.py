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

This example simulates a 1D thick Au film on Au substrate (no grating)
and compares the 0th-order reflectance with the analytical Fresnel result.
A thick film ensures T ~ 0, so we focus on R only.
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

    # --- Analytical Fresnel (normal incidence, air -> bulk Au) ---
    r_fresnel = (n_air - n_au) / (n_air + n_au)
    R_fresnel = abs(r_fresnel) ** 2
    print(f"\nAnalytical Fresnel (air -> Au): R = {R_fresnel:.6f}")

    # ------------------------------------------------------------------
    # Correct: ucell = conj(n), n_bot = raw n
    # ------------------------------------------------------------------
    n_au_conj = np.conj(n_au)
    ucell = np.array([[[n_au_conj]]])  # conjugated n in ucell

    mee = meent.call_mee(
        backend=0,
        pol=1,  # TM
        n_top=n_air,   # raw n
        n_bot=n_au,    # raw n — do NOT conjugate
        theta=0,
        fto=[0],       # no grating
        wavelength=wavelength,
        period=[100],  # arbitrary (uniform film)
        ucell=ucell,
        thickness=[200],  # thick Au — T is negligible
        type_complex=np.complex128,
    )
    result = mee.conv_solve()
    R = result.res.de_ri.sum()
    print(f"\nCorrect: ucell = conj(n_au), n_bot = n_au (raw)")
    print(f"  R = {R:.6f}  (Fresnel = {R_fresnel:.6f}, match = {abs(R - R_fresnel) < 1e-4})")

    # ------------------------------------------------------------------
    # WRONG: ucell not conjugated
    # ------------------------------------------------------------------
    mee.ucell = np.array([[[n_au]]])  # NOT conjugated
    result_w1 = mee.conv_solve()
    R_w1 = result_w1.res.de_ri.sum()
    print(f"\nWRONG: ucell = n_au (not conjugated)")
    print(f"  R = {R_w1:.6f}  (R > 1, energy conservation violated!)")



if __name__ == '__main__':
    run()
