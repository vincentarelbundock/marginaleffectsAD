# Estimation module for JAX-based marginal effects

from .utils import array as array

# Import submodules to make them accessible
from . import linear as linear
from . import logit as logit
from . import poisson as poisson
from . import probit as probit
