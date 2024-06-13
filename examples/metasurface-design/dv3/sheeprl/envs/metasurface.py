import os
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

# import deflector_gym
from deflector_gym.wrappers import BestRecorder
from deflector_gym.envs.meent_env import MeentIndexEfield

class SheepRLWrapper1(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env.observation_space = gym.spaces.Dict({
            "field": env.observation_space,
            "struct": gym.spaces.Box(low=-1., high=1., shape=(self.env.n_cells,), dtype=np.float32)
        }) 

    def reset(self, seed=42, options={}):
        obs, info = self.env.reset(seed, options)
        print(info)

        return {"field": obs, "struct": self.env.struct.copy()}, info
    
    def step(self, action):
        obs, rew, done, trunc, info = self.env.step(action)

        return {"field": obs, "struct": self.env.struct.copy()}, rew, done, trunc, info


class SheepRLWrapper2(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env.observation_space = gym.spaces.Dict({
            "real_field": gym.spaces.Box(low=0., high=255., shape=(1, 256, 256), dtype=np.float32),
            "imag_field": gym.spaces.Box(low=0., high=255., shape=(1, 256, 256), dtype=np.float32),
            "struct": gym.spaces.Box(low=-1., high=1., shape=(self.env.n_cells,), dtype=np.float32)
        }) 
        self.env.render_mode = 'rgb_array'
        self.field_traj = []
        self.struct_traj = []
        self.action_traj = []

    def _normalize(self, obs):
        vmax = 5.
        vmin = -vmax

        obs = np.clip(obs, vmin, vmax)
        obs = (obs - vmin) / (vmax - vmin)

        return obs

    def reset(self, seed=42, options={}):
        self.field_traj = []
        self.struct_traj = []
        self.action_traj = []

        obs, info = self.env.reset(seed, options)

        self.field_traj.append(obs)
        self.struct_traj.append(self.env.struct.copy())
        
        obs = self._normalize(obs)
        obs *= 255.
        print(info)
        self.rgb_array = obs[0][np.newaxis, :]

        return {
            "real_field": obs[0][np.newaxis, :],
            "imag_field": obs[1][np.newaxis, :],
            "struct": self.env.struct.copy(),
        }, info
    
    def step(self, action):
        obs, rew, done, trunc, info = self.env.step(action)
        self.field_traj.append(obs)
        self.struct_traj.append(self.env.struct.copy())
        self.action_traj.append(action)
        
        obs = self._normalize(obs)
        obs *= 255.
        self.rgb_array = obs[0][np.newaxis, :]

        return {
            "real_field": obs[0][np.newaxis, :],
            "imag_field": obs[1][np.newaxis, :],
            "struct": self.env.struct.copy()
        }, float(rew), done, trunc, info
    
    def render(self, mode='rgb_array'):
        return np.moveaxis(self.rgb_array.repeat(3, 0), 0, -1)

def get_deflector_env(id='MeentIndexEfield-v0'):
    env_id = 'MeentIndexEfield-v0'
    env_config = {'wavelength': 1100, 'desired_angle': 70, 'thickness': 325, 'n_cells': 256}
    env = MeentIndexEfield(**env_config)
    env = SheepRLWrapper2(env)
    # env = TimeLimit(env, max_episode_steps=128)
    # env = BestRecorder(env)
    
    return env

# envs = gym.vector.AsyncVectorEnv([get_deflector_env, get_deflector_env])
# print(envs.reset(seed=42, options={}))