from enum import Enum


class Action1D2(Enum):
    # Directional Action
    LEFT = 0
    NOOP = 1
    RIGHT = 2

class Action1D4(Enum):
    # Directional Action
    NOOP = 0
    LEFT_AIR = 1
    RIGHT_AIR = 2
    LEFT_SI = 3
    RIGHT_SI = 4

class Action1D22(Enum):
    # Directional Action
    NOOP = 0
    LEFT = 0
    RIGHT = 1
    SI = 2
    AIR = 3