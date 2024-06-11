import time

import deflector_gym


def test_fps():
    N = 100
    m_env = deflector_gym.make('MeentIndex-v0')
    m_env.reset()
    s = time.time()
    for i in range(N):
        m_env.step(m_env.action_space.sample())

    print(f'FPS: {N/(time.time() - s):0.3f}')
