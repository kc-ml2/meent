from jax import tree_util

from .rcwa import RCWAJax

tree_util.register_pytree_node(RCWAJax,
                               RCWAJax._tree_flatten,
                               RCWAJax._tree_unflatten)
