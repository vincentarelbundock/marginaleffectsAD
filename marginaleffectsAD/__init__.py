# Estimation module for JAX-based marginal effects

# Import submodules to make them accessible
from . import linear as linear
from . import glm as glm

# Re-export types for convenience
from .comparisons import ComparisonType as ComparisonType
from .glm.families import Family as Family, Link as Link
