import jax
import optax

from functools import partial

from tqdm import tqdm

from ..emsolver.rcwa import RCWAJax


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

    # @partial(jax.jit, static_argnums=(3, 4, 5))  # TODO: is self static? then what about self.conv_solver?
    # @jax.jit
    # def step(self, params, opt_state, optimizer, forward, loss_fn):
    #
    #     loss_value, grads = self._grad(params, forward, loss_fn)
    #
    #     updates, opt_state = optimizer.update(grads, opt_state, params)
    #     params = optax.apply_updates(params, updates)
    #     return params, opt_state, loss_value

    def fit(self, pois, forward, loss_fn, optimizer, iteration=1):
        params = {poi: (getattr(self, poi)) for poi in pois}
        opt_state = optimizer.init(params)

        # @partial(jax.jit, static_argnums=(0, 1, 2, 3))
        # def _fit(forward, loss_fn, optimizer, iteration):
        #     opt_state = optimizer.init(self.params)
        #
        #     for i in range(iteration):
        #         self.params, opt_state, loss_value = self.step(self.params, opt_state, optimizer, forward, loss_fn)
        #         if i % 1 == 0:
        #             jax.debug.print('step {}, loss: {}', i, loss_value)
        #     return self.params

        # @partial(jax.jit, static_argnums=(1, 2, 3, 4))
        @jax.jit
        def step(params, opt_state):
            loss_value, grads = self._grad(params, forward, loss_fn)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_value

        for _ in tqdm(range(iteration)):
            params, opt_state, loss_value = step(params, opt_state)

        [setattr(self, poi, params[poi]) for poi in pois]

        return params

