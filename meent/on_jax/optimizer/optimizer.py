import time
import numpy as np
import optax


import meent
import meent.testcase
from meent.on_jax.emsolver.rcwa import RCWAJax


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


class SGD(Grad):

    def __init__(self, parameters_to_fit, *args, **kwargs):
        super().__init__()
        self.parameters_to_fit = parameters_to_fit
        self.opt = torch.optim.SGD(parameters_to_fit, *args, **kwargs)

    def step(self):
        self.opt.step()

    def zero_grad(self):
        self.opt.zero_grad()


class OptimizerJax(RCWAJax, Grad):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def gradient_numerical(self):
        pass

    def fit(self, pois, forward, loss_fn, optimizer):
        [setattr(getattr(self, poi), 'requires_grad', True) for poi in pois]

        for i in range(1):
            optimizer.zero_grad()
            result = forward()  # Forward Prop.
            loss = loss_fn(result)  # Loss

            loss.backward()  # Back Prop.
            optimizer.step()
            print(2, self.ucell.grad)
        pass

    def fit_general(self, pois, forward, loss_fn, optimizer_algo, optimizer_kwargs):
        [setattr(getattr(self, poi), 'requires_grad', True) for poi in pois]

        obj_to_fit = [(getattr(self, poi)) for poi in pois]

        def call_optimizer(algorithm, obj_to_fit, *args, **kwargs):
            if algorithm.upper() == 'SGD':
                optimizer = SGD(obj_to_fit, *args, **kwargs)

            return optimizer

        optimizer = call_optimizer(optimizer_algo, obj_to_fit, **optimizer_kwargs)

        for i in range(1):
            optimizer.zero_grad()
            result = forward()  # Forward Prop.
            loss = loss_fn(result)  # Loss

            loss.backward()  # Back Prop.
            optimizer.step()
            print(2, self.ucell.grad)


if __name__ == '__main__':
    mode = 2
    dtype = 0
    device = 0

    conditions = meent.testcase.load_setting(mode, dtype, device)

    aa = OptimizerJax(**conditions)
    import meent.on_torch.optimizer.loss

    pois = ['ucell', 'thickness']
    parameters_to_fit = [(getattr(aa, poi)) for poi in pois]
    forward = aa.conv_solve
    loss_fn = meent.on_torch.optimizer.loss.LossDeflector(x_order=0, y_order=1)

    grad = aa.grad(pois, forward, loss_fn)
    print(1, grad)

    # case 1
    # opt = torch.optim.SGD(parameters_to_fit, lr=1E-2)
    opt = optax.sgd(learning_rate=1E-2)
    params = {'ucell': parameters_to_fit[0], 'thickness': parameters_to_fit[1]}
    opt_state = opt.init(params)

    compute_loss = lambda params, x, y: optax.l2_loss()


    def loss(params, batch, labels):

        forward()

        c_x = de_ti.shape[0] // 2
        c_y = de_ti.shape[1] // 2

        res = de_ti[c_x + self.x_order, c_y + self.y_order]

        loss_value = params['ucell']


    aa.fit(pois, forward, loss_fn, opt)
    print(3, grad)

    # case 2
    opt_algo = 'sgd'
    opt_kwargs = {'lr': 1E-2}
    aa.fit_general(pois, forward, loss_fn, opt_algo, opt_kwargs)
    print(3, grad)
