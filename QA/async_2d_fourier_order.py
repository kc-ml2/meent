import numpy as np

import meent

from meent.testcase import load_setting
from benchmarks.interface.Reticolo import Reticolo


mode = 0
dtype = 0
device = 0
grating_type = 2
pre = load_setting(mode, dtype, device, grating_type)

reti = Reticolo()
de_ri_top_inc, de_ti_top_inc, de_ri_bot_inc, de_ti_bot_inc = reti.run(**pre)
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
