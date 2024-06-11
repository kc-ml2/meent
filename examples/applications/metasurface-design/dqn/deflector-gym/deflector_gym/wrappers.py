import gym
import numpy as np

class BestRecorder(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.best = (0., None)  # efficiency, structure

    def step(self, action):
        # obs, rew, done, trunc, info = super().step(action)
        obs, rew, done, info = super().step(action)

        if self.eff > self.best[0]:
            self.best = (self.eff, self.struct.copy())

        info['max_eff'] = self.best[0]

        # return obs, rew, done, trunc, info
        return obs, rew, done, info

    def reset(self, *args, **kwargs):
        print(self.best[0])
        # obs, info = super().reset(*args, **kwargs)
        obs = super().reset(*args, **kwargs)
        info = {}
        info['max_eff'] = self.best[0]
        # self.best = (self.eff, self.struct.copy())

        # return obs, info
        return obs

class ExpandObservation(gym.Wrapper):
    def __init__(self, env):
        super(ExpandObservation, self).__init__(env)
        obs_space = self.observation_space
        self.observation_space = gym.spaces.Box(
            low=obs_space.low[0], high=obs_space.high[0],
            shape=(1, *obs_space.shape),
            dtype=np.float64
        )

    def step(self, action):
        obs, rew, done, info = super(ExpandObservation, self).step(action)
        obs = obs.reshape(1, -1)

        return obs, rew, done, info

    def reset(self, **kwargs):
        obs = super(ExpandObservation, self).reset(**kwargs)
        obs = obs.reshape(1, -1)

        return obs