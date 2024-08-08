from .modeler.modeling import ModelingJax
from .emsolver.rcwa import RCWAJax
from .optimizer.optimizer import OptimizerJax


class MeeJax(ModelingJax, RCWAJax, OptimizerJax):

    def __init__(self, *args, **kwargs):
        ModelingJax.__init__(self, *args, **kwargs)
        RCWAJax.__init__(self, *args, **kwargs)
        OptimizerJax.__init__(self, *args, **kwargs)

    def _tree_flatten(self):
        children = (self.n_top, self.n_bot, self.theta, self.phi, self.psi,
                    self.period, self.wavelength, self.ucell, self.thickness)
        aux_data = {
            'backend': self.backend,
            'pol': self.pol,
            'fto': self.fto,
            'ucell_materials': self.ucell_materials,
            'connecting_algo': self.connecting_algo,
            'perturbation': self.perturbation,
            'device': self.device,
            'type_complex': self.type_complex,
            'fourier_type': self.fourier_type,
            'enhanced_dfs': self.enhanced_dfs,
        }

        return children, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
