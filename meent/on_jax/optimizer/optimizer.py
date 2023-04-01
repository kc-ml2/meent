import jax
import optax

from tqdm import tqdm


class OptimizerJax:

    def __init__(self, *args, **kwargs):
        super().__init__()

    # def _tree_flatten(self):
    #     children = (self.n_I, self.n_II, self.theta, self.phi, self.psi,
    #                 self.period, self.wavelength, self.ucell, self.ucell_info_list, self.thickness)
    #     aux_data = {
    #         'backend': self.backend,
    #         'grating_type': self.grating_type,
    #         'pol': self.pol,
    #         'fourier_order': self.fourier_order,
    #         'ucell_materials': self.ucell_materials,
    #         'algo': self.algo,
    #         'perturbation': self.perturbation,
    #         'device': self.device,
    #         'type_complex': self.type_complex,
    #         'fft_type': self.fft_type,
    #     }
    #
    #     return children, aux_data

    # def _tree_flatten(self):
    #     children = ()
    #     aux_data = {}
    #
    #     return children, aux_data
    #
    # @classmethod
    # def _tree_unflatten(cls, aux_data, children):
    #     return cls(*children, **aux_data)

    @staticmethod
    def _grad(params, forward, loss_fn):

        def forward_pass(params, forward, loss):
            result = forward(**params)
            loss_value = loss(result)
            return loss_value

        loss_value, grads = jax.value_and_grad(forward_pass)(params, forward, loss_fn)
        return loss_value, grads

    def grad(self, pois, forward, loss_fn):
        params = {poi: (getattr(self, poi)) for poi in pois}
        _, grads = self._grad(params, forward, loss_fn)
        [setattr(self, poi, params[poi]) for poi in pois]

        return grads

    def fit(self, pois, forward, loss_fn, optimizer, iteration=1):
        params = {poi: (getattr(self, poi)) for poi in pois}
        opt_state = optimizer.init(params)

        @jax.jit
        def step(params, opt_state):
            loss_value, grads = self._grad(params, forward, loss_fn)
            grads = {k: v.conj() for k, v in grads.items()}
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_value

        for _ in tqdm(range(iteration)):
            params, opt_state, loss_value = step(params, opt_state)

        [setattr(self, poi, params[poi]) for poi in pois]

        return params

