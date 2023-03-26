import time

import jax
import jax.numpy as jnp

try:
    import jax.tools.colab_tpu
    jax.tools.colab_tpu.setup_tpu()
except:
    pass


# Google Drive mounting
gdrive_mounted = False
if gdrive_mounted:
    try:
        import drive.MyDrive.meent
    except:
        pass
else:
    import meent

from examples.ex_ucell import load_ucell, run_loop


gratings = [0, 1, 2]
backends = [0, 1, 2]
dtypes = [0, 1]
devices = [0]


# run_loop(gratings, backends, dtypes, devices)

gratings = [2]
backends = [1]
dtypes = [0]
devices = [0]

# with jax.default_device(jax.devices("cpu")[0]):
#     run_loop(gratings, backends, dtypes, [0])

# with jax.default_device(jax.devices("gpu")[0]):
#     run_loop(a, b, c, [1])
#
# with jax.default_device(jax.devices("cpu")[0]):
#     run_loop(a, b, c, [0])


# common
grating_type = 2
pol = 1  # 0: TE, 1: TM

n_I = 1  # n_incidence
n_II = 1  # n_transmission

theta = 10
phi = 0
psi = 0 if pol else 90

wavelength = 900

thickness = [500]
ucell_materials = [1, 3.48]
period = [100, 100]

fourier_order = 2
mode_options = {0: 'numpy', 1: 'JAX', 2: 'Torch', }

mode_key = 1
device = 2
type_complex = jnp.complex128

n_iter = 2

ucell = load_ucell(grating_type=grating_type)

AA = meent.call_mee(backend=mode_key, grating_type=grating_type, pol=pol, n_I=n_I, n_II=n_II, theta=theta, phi=phi,
                    psi=psi, fourier_order=fourier_order, wavelength=wavelength, period=period, ucell=ucell,
                    ucell_materials=ucell_materials,
                    thickness=thickness, device=device, type_complex=type_complex, fft_type='piecewise')

for i in range(n_iter):
    t0 = time.time()
    de_ri, de_ti = AA.conv_solve()
    print(f'run_cell: {i}: ', time.time() - t0)
