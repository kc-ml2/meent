import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'

import jax

import numpy as np
import jax.numpy as jnp

import time

size = 200
x = jnp.arange(size)
w = jnp.array([2., 3., 4.])


def convolve(x, w):
    output = []
    for i in range(1, len(x)-1):
        output.append(jnp.dot(x[i-1:i+2], w))
    return jnp.array(output)

iter = 2
##
for _ in range(iter):
    t0 = time.time()
    a=jax.jit(convolve)(x, w)
    print(time.time() - t0)
    print(a)


##
n_devices = jax.local_device_count()
xs = np.arange(size * n_devices).reshape(-1, size)
ws = np.stack([w] * n_devices)

##
for _ in range(iter):
    t0 = time.time()
    jax.vmap(convolve)(xs, ws)
    print(time.time() - t0)

##
for _ in range(iter):
    t0 = time.time()
    a = jax.pmap(convolve)(xs, ws).block_until_ready()
    print(time.time() - t0)

