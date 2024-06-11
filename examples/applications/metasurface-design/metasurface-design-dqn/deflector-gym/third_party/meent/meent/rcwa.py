import numpy as np


def call_solver(mode=0, *args, **kwargs):
    if mode == 0:
        from meent.on_numpy.rcwa import RCWALight
        RCWA = RCWALight(mode, *args, **kwargs)
    elif mode == 1:
        from meent.on_jax.rcwa import RCWAOpt
        RCWA = RCWAOpt(mode, *args, **kwargs)
    else:
        raise ValueError

    return RCWA


def sweep_wavelength(wls, *args, **kwargs):
    # wls = np.linspace(500, 1000, 10)
    spectrum_r = []
    spectrum_t = []
    # spectrum_r = np.zeros(wls.shape[0])
    # spectrum_t = np.zeros(wls.shape[0])

    for i, wl in enumerate(wls):
        wl = np.array([wl])
        solver = call_solver(wls=wl, *args, **kwargs)
        de_ri, de_ti = solver.run_ucell()
        spectrum_r.append(de_ri)
        spectrum_t.append(de_ti)
        # spectrum_r[i] = de_ri
        # spectrum_t[i] = de_ti
    spectrum_r = np.array(spectrum_r)
    spectrum_t = np.array(spectrum_t)
    return spectrum_r, spectrum_t
