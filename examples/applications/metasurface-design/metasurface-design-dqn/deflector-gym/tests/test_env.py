import deflector_gym

def test_instantiation():
    env = deflector_gym.make('MeentIndex-v0')
    print(env.reset())
    env = deflector_gym.make('MeentAction1D4-v0')
    print(env.reset())
    env = deflector_gym.make('ReticoloIndex-v0')
    print(env.reset())
