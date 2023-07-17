import torch
import numpy as np

from .modeler.modeling import ModelingTorch
from .emsolver.rcwa import RCWATorch
from .optimizer.optimizer import OptimizerTorch


class MeeTorch(ModelingTorch, RCWATorch, OptimizerTorch):

    def __init__(self, device=None, type_complex=None, *args, **kwargs):

        # device
        if device in (0, 'cpu'):
            self._device = torch.device('cpu')
        elif device in (1, 'gpu', 'cuda'):
            self._device = torch.device('cuda')
        elif type(device) is torch.device:
            self._device = device
        else:
            raise ValueError('device')

        # type_complex
        if type_complex in (0, torch.complex128, np.complex128):
            self._type_complex = torch.complex128
        elif type_complex in (1, torch.complex64, np.complex64):
            self._type_complex = torch.complex64
        else:
            raise ValueError('Torch type_complex')

        self._type_float = torch.float64 if self._type_complex is not torch.complex64 else torch.float32
        self._type_int = torch.int64 if self._type_complex is not torch.complex64 else torch.int32
        # self.perturbation = perturbation

        self.device = device
        self.type_complex = type_complex

        ModelingTorch.__init__(self, *args, **kwargs)
        RCWATorch.__init__(self, *args, **kwargs)
        OptimizerTorch.__init__(self, *args, **kwargs)
