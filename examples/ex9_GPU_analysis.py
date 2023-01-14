import time
import jax
import jax.numpy as jnp

import numpy as np

from jax import device_put

size = 500

from jax.config import config
# config.update("jax_enable_x64", True)

# jax.config.update('jax_platform_name', 'cpu')
# jax.default_device=jax.devices("cpu")
# print(jax.default_device)
config.update("jax_enable_x64", True)
with jax.default_device(jax.devices("cpu")[0]):
    aa = jnp.arange(size**2).reshape((size, size))
    t0 = time.time();[jnp.linalg.inv(aa) for _ in range(2000)];print(time.time() - t0)

with jax.default_device(jax.devices("gpu")[0]):
    aa = jnp.arange(size**2).reshape((size, size))
    # bb = device_put(aa)
    t0 = time.time();[jnp.linalg.inv(aa) for _ in range(2000)];print(time.time() - t0)

config.update("jax_enable_x64", False)
with jax.default_device(jax.devices("cpu")[0]):
    aa = jnp.arange(size**2).reshape((size, size))
    t0 = time.time();[jnp.linalg.inv(aa) for _ in range(2000)];print(time.time() - t0)

with jax.default_device(jax.devices("gpu")[0]):
    aa = jnp.arange(size**2).reshape((size, size))
    # bb = device_put(aa)
    t0 = time.time();[jnp.linalg.inv(aa) for _ in range(2000)];print(time.time() - t0)

print(1)

# t0 = time.time();[ee.inv(E_conv) for _ in range(10000)];print(time.time() - t0)

