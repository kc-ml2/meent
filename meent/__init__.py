try:
    import jax
    jax.config.update('jax_enable_x64', True)
except:
    pass

from .main import call_mee
