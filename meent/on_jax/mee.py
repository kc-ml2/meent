from .modeler.modeling import ModelingJax
from .emsolver.rcwa import RCWAJax
from .optimizer.optimizer import OptimizerJax


class MeeJax(ModelingJax, RCWAJax, OptimizerJax):

    def __init__(self, *args, **kwargs):
        ModelingJax.__init__(self, *args, **kwargs)
        RCWAJax.__init__(self, *args, **kwargs)
        OptimizerJax.__init__(self, *args, **kwargs)

    def _tree_flatten(self):
        children = (self.n_I, self.n_II, self.theta, self.phi,
                    self.period, self.wavelength, self.ucell, self.ucell_info_list, self.thickness)
        aux_data = {
            'backend': self.backend,
            'grating_type': self.grating_type,
            'pol': self.pol,
            'fourier_order': self.fourier_order,
            'ucell_materials': self.ucell_materials,
            'algo': self.algo,
            'perturbation': self.perturbation,
            'device': self.device,
            'type_complex': self.type_complex,
            'fft_type': self.fft_type,
        }

        return children, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
