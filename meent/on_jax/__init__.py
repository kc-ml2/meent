from jax import tree_util
from meent.on_jax.mee import MeeJax

tree_util.register_pytree_node(MeeJax,
                               MeeJax._tree_flatten,
                               MeeJax._tree_unflatten)
