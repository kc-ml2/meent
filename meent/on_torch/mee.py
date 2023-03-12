from .modeler.modeling import ModelingTorch
from .emsolver.rcwa import RCWATorch
from .optimizer.optimizer import OptimizerTorch


class MeeTorch(ModelingTorch, RCWATorch, OptimizerTorch):

    def __init__(self, *args, **kwargs):
        ModelingTorch.__init__(self, *args, **kwargs)
        RCWATorch.__init__(self, *args, **kwargs)
        OptimizerTorch.__init__(self, *args, **kwargs)
