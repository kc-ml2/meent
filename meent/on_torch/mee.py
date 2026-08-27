import torch
import numpy as np

from .modeler.modeling import ModelingTorch
from .emsolver.rcwa import RCWATorch
from .optimizer.optimizer import OptimizerTorch


class MeeTorch(ModelingTorch, RCWATorch, OptimizerTorch):
    """The object `meent.call_mee(backend=2)` returns: modeler, solver and optimizer in one.

    The three bases are kept separate because they are separate concerns - building a ucell,
    solving it, and differentiating through the solve - but a user holds a single object, so
    they are composed here rather than wired together by the caller.
    """

    def __init__(self, device=0, type_complex=0, *args, **kwargs):

        # device and dtype are resolved before the bases are constructed: each base reads
        # self._device / self._type_complex during its own __init__, so they have to exist
        # first. That is also why the bases are initialised by explicit calls instead of
        # super() - three of them need the same arguments, and the MRO would run only one.
        if device in (0, 'cpu'):
            self._device = torch.device('cpu')
        elif device in (1, 'gpu', 'cuda'):
            self._device = torch.device('cuda')
        elif type(device) is torch.device:
            self._device = device
        else:
            raise ValueError('device')
        if type_complex in (0, torch.complex128, np.complex128):
            self._type_complex = torch.complex128
        elif type_complex in (1, torch.complex64, np.complex64):
            self._type_complex = torch.complex64
        else:
            raise ValueError('Torch type_complex')
        # The real and integer dtypes are not independent choices: they follow the complex one
        # so that a float array and the complex array it feeds never mix precisions.
        self._type_float = torch.float64 if self._type_complex is not torch.complex64 else torch.float32
        self._type_int = torch.int64 if self._type_complex is not torch.complex64 else torch.int32
        # Redundant with the block above, and stricter: the `device` property in _base takes
        # only 0, 1 or a torch.device, so the 'cpu' / 'gpu' / 'cuda' spellings accepted three
        # lines up raise ValueError here. Passing those strings does not work today.
        self.device = device

        ModelingTorch.__init__(self, device=device, type_complex=type_complex, *args, **kwargs)
        RCWATorch.__init__(self, device=device, type_complex=type_complex, *args, **kwargs)
        OptimizerTorch.__init__(self, device=device, type_complex=type_complex, *args, **kwargs)
