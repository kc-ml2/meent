__version__ = '0.13.2'

try:
    import jax
    jax.config.update('jax_enable_x64', True)
except:
    pass

from .main import call_mee
from .on_numpy.modeler.modeling import list_materials, print_materials
