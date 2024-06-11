import gym
from gym.envs.registration import register

def make(*args, **kwargs):
    return gym.make(*args, **kwargs)

try:
    register(
        id='ReticoloIndex-v0',
        entry_point='deflector_gym.envs.reticolo_env:ReticoloIndex',
    )
except Exception as e:
    raise Warning(f'Reticolo environments not available\n{e}')

register(
    id='MeentIndex-v0',
    entry_point='deflector_gym.envs.meent_env:MeentIndex',
)

register(
    id='MeentAction1D2-v0',
    entry_point='deflector_gym.envs.meent_env:MeentAction1D2'
)

register(
    id='MeentAction1D4-v0',
    entry_point='deflector_gym.envs.meent_env:MeentAction1D4'
)

