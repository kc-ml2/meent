try:
    import jax
    jax.config.update('jax_enable_x64', True)
except:
    pass


from jax import tree_util

from .modeling import ModelingJax

tree_util.register_pytree_node(ModelingJax,
                               ModelingJax._tree_flatten,
                               ModelingJax._tree_unflatten)
