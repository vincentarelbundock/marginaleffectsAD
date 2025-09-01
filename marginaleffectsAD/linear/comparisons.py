import jax
import jax.numpy as jnp
import numpy as np
from jax import jacrev, jit, lax
from typing import Callable, Optional
from ..comparisons import ComparisonType, _compute_comparison
from ..utils import (
    group_reducer,
    create_jacobian,
    create_jacobian_byT,
    create_jacobian_byG,
)


def _comparison_core(
    beta: jnp.ndarray,
    X_hi: jnp.ndarray,
    X_lo: jnp.ndarray,
    comparison_type: int,
) -> jnp.ndarray:
    """Core comparison function for linear models - single source of truth."""
    pred_hi = X_hi @ beta
    pred_lo = X_lo @ beta
    return _compute_comparison(comparison_type, pred_hi, pred_lo)


@jit
def _comparison(
    beta: jnp.ndarray, X_hi: jnp.ndarray, X_lo: jnp.ndarray, comparison_type: int
) -> jnp.ndarray:
    return _comparison_core(beta, X_hi, X_lo, comparison_type)


@jit
def _comparison_byT(
    beta: jnp.ndarray, X_hi: jnp.ndarray, X_lo: jnp.ndarray, comparison_type: int
) -> jnp.ndarray:
    comp = _comparison_core(beta, X_hi, X_lo, comparison_type)
    return jnp.mean(comp)


def comparison_byG(
    beta: jnp.ndarray,
    X_hi: jnp.ndarray,
    X_lo: jnp.ndarray,
    groups: jnp.ndarray,
    num_groups: int,
    comparison_type: int,
) -> jnp.ndarray:
    comp = _comparison_core(beta, X_hi, X_lo, comparison_type)
    return group_reducer(comp, groups, num_groups)




# Public comparison functions
comparison = _comparison
comparison_byT = _comparison_byT


# Create jacobian functions using factory functions
jacobian = create_jacobian(_comparison)
jacobian_byT = create_jacobian_byT(_comparison_byT)
jacobian_byG = create_jacobian_byG(comparison_byG)


# Note: *_byG functions cannot be JIT compiled due to num_groups parameter
