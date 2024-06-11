import os
from functools import partial
from pathlib import Path

import gym
import numpy as np

from .base import DeflectorBase
from .actions import Action1D2

try:
    import matlab.engine
except:
    raise Warning(
        'matlab python API not installed, '
        'try installing pip install matlabengine=={YOUR MATLAB VERSION}'
    )

RETICOLO_MATLAB = os.path.join(Path(__file__).parent.parent.parent.absolute(), 'third_party/reticolo_allege')
SOLVER_MATLAB = os.path.join(Path(__file__).parent.parent.parent.absolute(), 'third_party/solvers')


class MatlabBase(DeflectorBase):
    def __init__(
            self,
            n_cells=256,
            wavelength=1100,
            desired_angle=70
    ):
        super().__init__(n_cells, wavelength, desired_angle)

        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.eng.genpath(RETICOLO_MATLAB))
        self.eng.addpath(self.eng.genpath(SOLVER_MATLAB))
        self.wavelength_mtl = matlab.double([wavelength])
        self.desired_angle_mtl = matlab.double([desired_angle])

    def get_efficiency(self, struct: np.array):
        return self.eng.Eval_Eff_1D(
            matlab.double(struct.tolist()),
            self.wavelength_mtl,
            self.desired_angle_mtl
        )

    def close(self):
        # recommend to close env when done
        self.eng.quite()

class ReticoloIndex(MatlabBase):
    """
    legacy env for compatibility with chaejin's UNet
    will move to meent later on
    """
    def __init__(
            self,
            n_cells=256,
            wavelength=1100,
            desired_angle=70,
            *args,
            **kwargs
    ):
        super().__init__(n_cells, wavelength, desired_angle)

        self.observation_space = gym.spaces.Box(
            low=-1., high=1.,
            shape=(n_cells,),
            dtype=np.float64
        )
        self.action_space = gym.spaces.Discrete(n_cells)

    def reset(self):
        self.struct = self.initialize_struct()
        self.eff = self.get_efficiency(self.struct)

        return self.struct.copy()  # for 1 channel

    def step(self, action):
        prev_eff = self.eff

        self.flip(action)
        self.eff = self.get_efficiency(self.struct)

        reward = self.eff - prev_eff

        # unsqueeze for 1 channel
        return self.struct.copy(), reward, False, {}

