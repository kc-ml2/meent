import numpy as np
import gym

from .utils import random_bunch_init


class DeflectorBase(gym.Env):
    """
    This is interface class for deflection simulation
    You will need to inherit this interface and,
    implement abstract methods such as get_efficiency,
    define observation spaces, etc
    """
    def __init__(
            self,
            n_cells=256,
            wavelength=1100,
            desired_angle=70,
            order=40,
            thickness=325,
    ):
        self.thickness = thickness
        self.order = order
        self.n_cells = n_cells
        self.wavelength = wavelength
        self.desired_angle = desired_angle
        self.struct = None
        self.eff = None  # uninitialized

    def initialize_struct(self, *args, **kwargs):
        # default initialization is genetic algorithm(ga)
        return random_bunch_init(*args, **kwargs)

    def get_efficiency(self, struct: np.array) -> float:
        raise NotImplementedError

    def flip(self, pos):
        if 0 <= pos <= (self.n_cells - 1):
            self.struct[pos] = 1 if self.struct[pos] == -1 else -1
        else:
            # if out of boundary, do nothing
            # the agent will learn the boundary
            pass