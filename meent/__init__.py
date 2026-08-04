__version__ = '0.13.2'

try:
    import jax
    jax.config.update('jax_enable_x64', True)
except:
    pass

from .main import call_mee
from .dispersion import graphene_falkovsky
# Material lookup is duplicated per backend, but the three agree on the value they return: the
# table itself is plain data, and find_nk_index gives the same n - ik on all of them. Re-export
# one set so callers do not have to reach into a backend module to read an index.
from .on_numpy.modeler.modeling import (find_nk_index, list_materials, print_materials,
                                        read_material_table)
