# Estimation module for JAX-based marginal effects

# import JAX with 64bit precision (#1)
# https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision
import jax
jax.config.update("jax_enable_x64", True)

from .utils import array as array

# Import submodules to make them accessible
from . import linear as linear
from . import glm as glm
from . import comparisons as comparisons
