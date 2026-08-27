from tqdm import tqdm

from ..emsolver.rcwa import RCWATorch


class Grad:
    def __init__(self):
        pass


class OptimizerTorch(Grad):
    """Differentiation helpers mixed into MeeTorch.

    `pois` - parameters of interest - are the *names* of solver attributes to differentiate
    with respect to, e.g. ['ucell', 'thickness']. Naming them rather than passing the tensors
    lets `forward` be a plain zero-argument closure that reads the current values off self,
    which is what makes the same call work for a gradient and for a fitting loop.
    """

    def __init__(self, *args, **kwargs):

        super().__init__()

    def grad(self, pois, forward, loss_fn):
        """One backward pass. Returns {name: gradient}.

        Note the gradients accumulate: torch adds into .grad, and nothing zeroes it here, so
        calling this twice without clearing gives the sum of the two. `fit` zeroes explicitly
        for that reason.
        """
        [setattr(getattr(self, poi), 'requires_grad', True) for poi in pois]
        result = forward()
        loss = loss_fn(result)
        loss.backward()
        grad = {poi: getattr(self, poi).grad for poi in pois}

        return grad

    def meent_optimizer(self, _pois, _opt, *args, **kwargs):
        # `_opt` is the optimizer class, not an instance - it cannot be built before this
        # point because it needs the very tensors the names resolve to.
        _parameters_to_fit = [(getattr(self, poi)) for poi in _pois]
        res = _opt(_parameters_to_fit, *args, **kwargs)
        return res

    def fit(self, pois, forward, loss_fn, optimizer, opt_options, iteration=1):
        """Optimise the named attributes in place. Returns them after the last step.

        The optimizer holds references to the tensors, and it updates them in place, so the
        solver sees the new values on the next `forward()` without anything being reassigned.
        """
        optimizer = self.meent_optimizer(pois, optimizer, **opt_options)
        [setattr(getattr(self, poi), 'requires_grad', True) for poi in pois]

        for _ in tqdm(range(iteration)):
            optimizer.zero_grad()
            result = forward()
            loss_value = loss_fn(result)

            loss_value.backward()
            optimizer.step()

        return [getattr(self, poi) for poi in pois]
