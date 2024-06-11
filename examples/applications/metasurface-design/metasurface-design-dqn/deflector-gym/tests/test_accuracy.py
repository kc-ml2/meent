import numpy as np
from tqdm import tqdm

import deflector_gym

def test_compare_random_structure():
    N = 100
    print(f'comparing random {N} structures')
    for _ in tqdm(range(N)):
        test_struct = np.random.choice([-1, 1], 256)
        m_env = deflector_gym.make('MeentIndex-v0')
        m_eff = m_env.get_efficiency(test_struct)
        print(m_eff)
        r_env = deflector_gym.make('ReticoloIndex-v0')
        r_eff = r_env.get_efficiency(test_struct)
        print(f'{m_eff} {r_eff}')

        np.testing.assert_almost_equal(m_eff, r_eff)
    r_eff.close()
