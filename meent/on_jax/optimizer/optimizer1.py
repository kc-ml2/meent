import random
from typing import Tuple

import optax
import jax.numpy as jnp
import jax
import numpy as np

BATCH_SIZE = 5
NUM_TRAIN_STEPS = 1_000
RAW_TRAINING_DATA = np.random.randint(255, size=(NUM_TRAIN_STEPS, BATCH_SIZE, 1))

TRAINING_DATA = np.unpackbits(RAW_TRAINING_DATA.astype(np.uint8), axis=-1)
LABELS = jax.nn.one_hot(RAW_TRAINING_DATA % 2, 2).astype(jnp.float32).reshape(NUM_TRAIN_STEPS, BATCH_SIZE, 2)

initial_params = {
    'hidden': jax.random.normal(shape=[8, 32], key=jax.random.PRNGKey(0)),
    'output': jax.random.normal(shape=[32, 2], key=jax.random.PRNGKey(1)),
}


def net_(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    x = jnp.dot(x, params['hidden'])
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['output'])
    return x


def net(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    x = jnp.dot(x, params['hidden'])
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['output'])
    return x


def loss(params: optax.Params, batch: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    y_hat = net(batch, params)

    # optax also provides a number of common loss functions.
    loss_value = optax.sigmoid_binary_cross_entropy(y_hat, labels).sum(axis=-1)

    return loss_value.mean()


def fit(params: optax.Params, optimizer: optax.GradientTransformation) -> optax.Params:
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, batch, labels):
        loss_value, grads = jax.value_and_grad(loss)(params, batch, labels)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    for i, (batch, labels) in enumerate(zip(TRAINING_DATA, LABELS)):
        params, opt_state, loss_value = step(params, opt_state, batch, labels)
        if i % 100 == 0:
            print(f'step {i}, loss: {loss_value}')

    return params


# Finally, we can fit our parametrized function using the Adam optimizer
# provided by optax.
optimizer = optax.adam(learning_rate=1e-2)
params = fit(initial_params, optimizer)


if __name__ == '__main__':
    import meent.testcase

    mode = 1
    dtype = 0
    device = 0
    grating_type = 2

    conditions = meent.testcase.load_setting(mode, dtype, device, grating_type)

    initial_params = {
        # 'hidden': jax.random.normal(shape=[8, 32], key=jax.random.PRNGKey(0)),
        # 'output': jax.random.normal(shape=[32, 2], key=jax.random.PRNGKey(1)),
        'ucell': conditions['ucell']
    }

    optimizer = optax.adam(learning_rate=1e-2)
    params = fit(initial_params, optimizer)

    print(1)
    #######################
    aa = OptimizerTorch(**conditions)
    import meent.on_torch.optimizer.loss

    pois = ['ucell', 'thickness']
    parameters_to_fit = [(getattr(aa, poi)) for poi in pois]
    forward = aa.conv_solve
    loss_fn = meent.on_torch.optimizer.loss.LossDeflector(x_order=0, y_order=1)

    grad = aa.grad(pois, forward, loss_fn)
    print(1, grad)

    # case 1
    opt = torch.optim.SGD(parameters_to_fit, lr=1E-2)
    aa.fit(pois, forward, loss_fn, opt)
    print(3, grad)

    # case 2
    opt_algo = 'sgd'
    opt_kwargs = {'lr': 1E-2}
    aa.fit_general(pois, forward, loss_fn, opt_algo, opt_kwargs)
    print(3, grad)
