import numpy as np

import meent

from meent.testcase import load_setting
from benchmarks.interface.Reticolo import Reticolo


# backend = 0
# dtype = 0
# device = 0
# grating_type = 2
# pre = load_setting(backend, dtype, device, grating_type)


condition = {}

condition['grating_type'] = 0  # 0 : just 1D grating, 1 : 1D rotating grating, 2 : 2D grating

condition['pol'] = 0  # 0: TE, 1: TM

condition['n_I'] = 1  # n_incidence
condition['n_II'] = 1.5  # n_transmission

condition['theta'] = 0 * np.pi / 180
condition['phi'] = 0 * np.pi / 180
condition['psi'] = 0 if condition['pol'] else 90 * np.pi / 180

# condition['fourier_order'] = 50
# condition['period'] = [1000.]
# condition['wavelength'] = 900
# condition['thickness'] = [200., 200., 200., 200., 200., 500]
#
# ucell = np.array(
#         [
#             0, 0, 1, 1, 1, 0, 0, 1, 1, 1,
#             1, 0, 0, 1, 0, 0, 0, 1, 1, 1,
#             1, 1, 0, 1, 1, 1, 1, 0, 1, 1,
#             1, 1, 1, 0, 1, 0, 0, 1, 1, 1,
#             0, 0, 0, 0, 1, 0, 0, 1, 1, 1,
#             0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#         ], dtype=np.float64).reshape((len(condition['thickness']), -1, 10)) * 1.75 + 2.25


condition['fourier_order'] = 40
condition['period'] = [1000.]
condition['wavelength'] = [651. - 100. * wave_units for wave_units in range(3)]
condition['thickness'] = [200., 200., 200., 200., 200., 500]  # final term is for h_substrate

n_1 = 1.5
n_2 = 2

cell_per_period = 10

ucell = np.array(
        [
            # 0, 0, 1, 1, 1, 0, 0, 1, 1, 1,
            # 1, 0, 0, 1, 0, 0, 0, 1, 1, 1,
            # 1, 1, 0, 1, 1, 1, 1, 0, 1, 1,
            # 1, 1, 1, 0, 1, 0, 0, 1, 1, 1,
            # 0, 0, 0, 0, 1, 0, 0, 1, 1, 1,
            0, 0, 1, 1, 1, 0, 0, 1, 1, 1,
            1, 0, 0, 1, 0, 0, 0, 1, 1, 1,
            1, 1, 0, 1, 1, 1, 1, 0, 1, 1,
            1, 1, 1, 0, 1, 0, 0, 1, 1, 1,
            0, 0, 0, 0, 1, 0, 0, 1, 1, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ], dtype=np.float64).reshape((len(condition['thickness']), -1, cell_per_period)) * (n_2 - n_1) + n_1

condition['ucell'] = ucell
condition['ucell'] = np.repeat(ucell, 10, axis=2)





reti = Reticolo()
de_ri_top_inc, de_ti_top_inc, de_ri_bot_inc, de_ti_bot_inc = reti.run_res2(**condition)
de_ri_reti = de_ri_top_inc.flatten()
de_ti_reti = de_ti_top_inc.flatten()

print('reti de_ri:', de_ri_reti)
print('reti de_ti:', de_ti_reti)

# Numpy
mode = 0
pre = load_setting(mode, dtype, device, grating_type)
mee = meent.call_mee(backend=mode, perturbation=1E-30, **pre)
mee.fft_type = 0

de_ri, de_ti = mee.conv_solve()
center = np.array(de_ri.shape) // 2
de_ri_cut = de_ri[center[0] - 1:center[0] + 2, center[1] - 1:center[1] + 2]
de_ti_cut = de_ti[center[0] - 1:center[0] + 2, center[1] - 1:center[1] + 2]
cut_index = [3, 1, 4, 7, 5]
# cut_index = [3, 1, 4, 5]
de_ri_cut = de_ri_cut.flatten()[cut_index]
de_ti_cut = de_ti_cut.flatten()[cut_index]

print('Norm(Reti, NPY): ', np.linalg.norm(de_ri_reti - de_ri_cut), np.linalg.norm(de_ti_reti - de_ti_cut))

# JAX
mode = 1
pre = load_setting(mode, dtype, device, grating_type)
mee = meent.call_mee(backend=mode, perturbation=1E-30, **pre)
mee.fft_type = 0

de_ri, de_ti = mee.conv_solve()
center = np.array(de_ri.shape) // 2

de_ri, de_ti = np.array(de_ri), np.array(de_ti)
de_ri_cut = de_ri[center[0] - 1:center[0] + 2, center[1] - 1:center[1] + 2]
de_ti_cut = de_ti[center[0] - 1:center[0] + 2, center[1] - 1:center[1] + 2]
de_ri_cut = de_ri_cut.flatten()[cut_index]
de_ti_cut = de_ti_cut.flatten()[cut_index]

# print('meen jx de_ri:', de_ri_cut)
# print('meen jx de_ti:', de_ti_cut)

print('Norm(Reti, JAX): ', np.linalg.norm(de_ri_reti - de_ri_cut), np.linalg.norm(de_ti_reti - de_ti_cut))

# Torch
mode = 2
pre = load_setting(mode, dtype, device, grating_type)
mee = meent.call_mee(backend=mode, perturbation=1E-30, **pre)
mee.fft_type = 0

de_ri, de_ti = mee.conv_solve()
center = np.array(de_ri.shape) // 2

de_ri, de_ti = np.array(de_ri), np.array(de_ti)
de_ri_cut = de_ri[center[0] - 1:center[0] + 2, center[1] - 1:center[1] + 2]
de_ti_cut = de_ti[center[0] - 1:center[0] + 2, center[1] - 1:center[1] + 2]
de_ri_cut = de_ri_cut.flatten()[cut_index]
de_ti_cut = de_ti_cut.flatten()[cut_index]

# print('meen to de_ri:', de_ri_cut)
# print('meen to de_ti:', de_ti_cut)

print('Norm(Reti, TOR): ', np.linalg.norm(de_ri_reti - de_ri_cut), np.linalg.norm(de_ti_reti - de_ti_cut))
