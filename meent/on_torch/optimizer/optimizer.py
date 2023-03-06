from ..emsolver.rcwa import RCWATorch


class Grad:
    def __init__(self):
        pass

    def grad(self, pois, forward, loss_fn):
        [setattr(getattr(self, poi), 'requires_grad', True) for poi in pois]
        result = forward()  # Forward Prop.
        loss = loss_fn(result)  # Loss
        loss.backward()  # Back Prop.
        grad = {poi: getattr(self, poi).grad for poi in pois}  # gradient

        return grad


class OptimizerTorch(RCWATorch, Grad):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def gradient_numerical(self):
        pass

    def meent_optimizer(self, _pois, _opt, *args, **kwargs):
        _parameters_to_fit = [(getattr(self, poi)) for poi in _pois]
        res = _opt(_parameters_to_fit, *args, **kwargs)
        return res

    def fit(self, pois, forward, loss_fn, optimizer, opt_options, iteration=100):
        optimizer = self.meent_optimizer(pois, optimizer, **opt_options)
        [setattr(getattr(self, poi), 'requires_grad', True) for poi in pois]

        for i in range(iteration):
            optimizer.zero_grad()
            result = forward()  # Forward Prop.
            loss_value = loss_fn(result)  # Loss

            loss_value.backward()  # Back Prop.
            optimizer.step()
            print(f'step {i}, loss: {loss_value}')
