import time
import jax
from jax import jit
import numpy as np
import jax.numpy as jnp

import torch

def jit_vs_nonjit():
    ucell = jnp.zeros((1, 100, 100))

    res = jnp.zeros(ucell.shape, dtype='complex')

    @jit
    def assign(arr, index, value):
        arr = arr.at[index].set(value)
        return arr

    assign_index = (0, 0, 0)
    assign_value = 3 ** 2

    t0 = time.time()
    arr = res.at[assign_index].set(assign_value)
    print('at set 1: ', time.time() - t0)

    t0 = time.time()
    arr = res.at[assign_index].set(assign_value)
    print('at set 2: ', time.time() - t0)

    t0 = time.time()
    arr = res.at[assign_index].set(assign_value)
    print('at set 3: ', time.time() - t0)

    t0 = time.time()
    arr = assign(res, assign_index, assign_value).block_until_ready()
    print('assign 1: ', time.time() - t0)

    t0 = time.time()
    arr = assign(res, assign_index, assign_value).block_until_ready()
    print('assign 2: ', time.time() - t0)


    t0 = time.time()
    arr = assign(res, assign_index, assign_value)
    print('assign 3: ', time.time() - t0)


    for i in range(1):
        # res = assign(res, assign_index, assign_value)
        arr = res.at[assign_index].set(assign_value)
    print(time.time() - t0)
    t0 = time.time()
    for i in range(100):
        # res = assign(res, assign_index, assign_value)
        arr = res.at[assign_index].set(assign_value)
    print(time.time() - t0)

    t0 = time.time()
    for i in range(1):
        arr = assign(res, assign_index, assign_value)
        # arr = res.at[tuple(assign_index)].set(assign_value)
    print(time.time() - t0)

    t0 = time.time()
    for i in range(100):
        arr = assign(res, assign_index, assign_value).block_until_ready()
        # arr = res.at[tuple(assign_index)].set(assign_value)
    print(time.time() - t0)

    # Result

    # at set 1:  0.03652310371398926
    # at set 2:  0.0010008811950683594
    # at set 3:  0.0007517337799072266
    # assign 1:  0.016371965408325195
    # assign 2:  4.601478576660156e-05
    # assign 3:  3.0994415283203125e-05

    # at set 1:        0.0009369850158691406
    # at set 2 to 102: 0.06914997100830078
    # assign 1:        5.412101745605469e-05
    # assign 2 to 102: 0.0008990764617919922


def test():
    ss = 4000
    aa = np.arange(ss*ss).reshape((ss, ss))
    bb = torch.Tensor(aa)
    itera = 1000

    for _ in range(itera):
        t0 = time.time()
        np.linalg.eig(aa)
        print(time.time() - t0)

    print('jax')
    for _ in range(itera):
        t0 = time.time()
        jnp.linalg.eig(aa)
        print(time.time() - t0)

    print('jit')
    t0 = time.time()
    eig = jax.jit(jnp.linalg.eig)
    eig(aa)
    print(time.time() - t0)

    for _ in range(itera-1):
        t0 = time.time()
        eig(aa)
        print(time.time() - t0)

    print('torch')
    for _ in range(itera):
        t0 = time.time()
        torch.linalg.eig(bb)
        print(time.time()-t0)



if __name__ == '__main__':
    # Global flag to set a specific platform, must be used at startup.
    jax.config.update('jax_platform_name', 'cpu')

    x = jnp.square(2)
    print(repr(x.device_buffer.device()))  # CpuDevice(id=0)

    # jit_vs_nonjit()
    test()