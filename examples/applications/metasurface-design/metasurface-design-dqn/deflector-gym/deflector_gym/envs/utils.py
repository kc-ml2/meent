import time

import gym
import numpy as np


class DummyEnv(gym.Env):
    def __init__(self, **kwargs):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-1., high=1.,
            shape=(1, kwargs['n_cells']),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(kwargs['n_cells'])

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        time.sleep(0.01)

        return self.observation_space.sample(), 0.1, False, {}

    def render(self):
        pass


def random_bunch_init(n_cells=256, mfs=30, mu=0) -> np.array:
    """Genetic Algorithm
    TODO: explain detail
    """
    sigma = n_cells / 4
    # Use This!!!!!!!
    i = mfs + 1
    # alloc = 1.0
    img = [1.] * n_cells
    while i < n_cells:
        # alloc = alloc * (-1)
        temp = i + mfs + int(abs(mu + np.random.rand() * sigma))
        img[i] = -1.
        i = temp + 1

    return np.array(img)
