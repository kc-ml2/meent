"""Example: Extracting reflection/transmission phase from meent results.

conv_solve() returns a Result object containing complex Rayleigh
coefficients (R_s, R_p, T_s, T_p) in addition to real-valued
diffraction efficiencies (de_ri, de_ti).

The complex coefficients carry phase information, which is essential
for metasurface design and wavefront engineering applications.

Access pattern:
    result = mee.conv_solve()
    R_s = result.res.R_s      # complex array
    phase = np.angle(R_s)     # phase in radians

For 1D gratings, R_s has shape (1, 2*fto+1) where the 0th order
is at index [:, fto].
"""

import numpy as np
import meent


def run():
    wavelength = 900  # nm
    n_si = 3.48
    n_air = 1.0
    fto = 10

    # 1D Si grating on Si substrate
    nx = 100
    ucell = np.ones((1, 1, nx)) * n_air
    ucell[0, 0, 30:70] = n_si  # 40% fill factor

    mee = meent.call_mee(
        backend=0,
        pol=0,  # TE
        n_top=n_air,
        n_bot=n_si,
        theta=0,
        fto=[fto],
        wavelength=wavelength,
        period=[500],
        ucell=ucell,
        thickness=[200],
        type_complex=np.complex128,
    )

    result = mee.conv_solve()

    # --- Diffraction efficiencies (real) ---
    de_ri = result.res.de_ri  # shape: (1, 2*fto+1)

    # --- Complex Rayleigh coefficients ---
    R_s = result.res.R_s  # complex reflection amplitude (TE)
    T_s = result.res.T_s  # complex transmission amplitude (TE)

    # 0th order: index [0, fto] for 1D
    r0 = R_s[0, fto]
    t0 = T_s[0, fto]

    print("=== 1D Si grating, TE, normal incidence ===")
    print(f"Wavelength: {wavelength} nm, Period: 500 nm, Thickness: 200 nm")
    print(f"\n0th order reflection:")
    print(f"  |r0|^2 = {abs(r0)**2:.6f}  (de_ri = {de_ri[0, fto]:.6f})")
    print(f"  phase  = {np.degrees(np.angle(r0)):+.2f} deg")
    print(f"\n0th order transmission:")
    print(f"  |t0|^2 = {abs(t0)**2:.6f}")
    print(f"  phase  = {np.degrees(np.angle(t0)):+.2f} deg")

    # --- Phase vs grating thickness ---
    print("\n=== Reflection phase vs grating thickness ===")
    print(f"{'Thickness (nm)':>15s} {'Phase (deg)':>12s} {'|r0|^2':>10s}")
    for t in [50, 100, 150, 200, 250, 300]:
        mee.thickness = [t]
        res = mee.conv_solve()
        r = res.res.R_s[0, fto]
        print(f"{t:>15d} {np.degrees(np.angle(r)):>+12.2f} {abs(r)**2:>10.6f}")


if __name__ == '__main__':
    run()
