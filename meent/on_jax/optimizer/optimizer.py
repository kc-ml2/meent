import jax
import optax

from functools import partial

from ..emsolver.rcwa import RCWAJax


class Grad:

    @staticmethod
    def forward(params, forward, loss):
        result = forward(**params)
        loss_value = loss(result)
        return loss_value

    def _grad(self, params, forward, loss_fn):
        loss_value, grads = jax.value_and_grad(self.forward)(params, forward, loss_fn)
        return loss_value, grads

    def grad(self, pois, forward, loss_fn):
        params = {poi: (getattr(self, poi)) for poi in pois}
        loss_value, grads = jax.value_and_grad(self.forward)(params, forward, loss_fn)
        return loss_value, grads


class OptimizerJax(RCWAJax, Grad):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def _tree_flatten(self):  # TODO: check args and kwargs
        children = (self.n_I, self.n_II, self.theta, self.phi, self.psi,
                    self.period, self.wavelength, self.ucell, self.thickness)
        aux_data = {
            'mode': self.mode,
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

    def gradient_numerical(self):
        pass

    @partial(jax.jit, static_argnums=(3, 4, 5))
    def step(self, params, opt_state, optimizer,  forward, loss_fn):

        loss_value, grads = self._grad(params, forward, loss_fn)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    def fit(self, pois, forward, loss_fn, optimizer, iteration=100):
        params = {poi: (getattr(self, poi)) for poi in pois}

        opt_state = optimizer.init(params)

        for i in range(iteration):
            params, opt_state, loss_value = self.step(params, opt_state, optimizer, forward, loss_fn)
            if i % 1 == 0:
                print(f'step {i}, loss: {loss_value}')
