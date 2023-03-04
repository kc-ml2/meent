from jax import tree_util

from .optimizer import OptimizerJax

tree_util.register_pytree_node(OptimizerJax,
                               OptimizerJax._tree_flatten,
                               OptimizerJax._tree_unflatten)
