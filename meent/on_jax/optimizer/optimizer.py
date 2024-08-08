import jax

try:
    import optax
except TypeError as e:
    import warnings

    warnings.warn('Importing optax failed. You can run RCWA but not optimization in JaxMeent. '
                  'One possible reason is python version: optax support python>=3.9.')

from tqdm import tqdm


class OptimizerJax:

    def __init__(self, *args, **kwargs):
        super().__init__()

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
