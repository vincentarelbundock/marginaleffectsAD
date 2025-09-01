import jax
import jax.numpy as jnp
import numpy as np
from jax import jacrev, jit
from typing import Callable, Optional, Union
from .families import Family, Link, linkinv, get_default_link
from ..comparisons import ComparisonType, _compute_comparison
from ..utils import (
    group_reducer,
    create_jacobian,
    create_jacobian_byT,
    create_jacobian_byG,
)


def _resolve_link(family_type: int, link_type: Optional[int]) -> int:
    """Resolve link type, using default if None."""
    return link_type if link_type is not None else get_default_link(family_type)


def _comparison_core(
    beta: jnp.ndarray,
    X_hi: jnp.ndarray,
    X_lo: jnp.ndarray,
    comparison_type: int,
    family_type: int,
    link_type: Optional[int],
) -> jnp.ndarray:
    """Core comparison function - single source of truth for comparison vector computation."""
    lt = _resolve_link(family_type, link_type)
    pred_hi = linkinv(lt, X_hi @ beta)
    pred_lo = linkinv(lt, X_lo @ beta)
    return _compute_comparison(comparison_type, pred_hi, pred_lo)


@jit
def _comparison(
    beta: jnp.ndarray,
    X_hi: jnp.ndarray,
    X_lo: jnp.ndarray,
    comparison_type: int,
    family_type: int,
    link_type: int = None,
) -> jnp.ndarray:
    return _comparison_core(beta, X_hi, X_lo, comparison_type, family_type, link_type)


@jit
def _comparison_byT(
    beta: jnp.ndarray,
    X_hi: jnp.ndarray,
    X_lo: jnp.ndarray,
    comparison_type: int,
    family_type: int,
    link_type: int = None,
) -> jnp.ndarray:
    comp = _comparison_core(beta, X_hi, X_lo, comparison_type, family_type, link_type)
    return jnp.mean(comp)


def comparison_byG(
    beta: jnp.ndarray,
    X_hi: jnp.ndarray,
    X_lo: jnp.ndarray,
    groups: jnp.ndarray,
    num_groups: int,
    comparison_type: int,
    family_type: int,
    link_type: int = None,
) -> jnp.ndarray:
    comp = _comparison_core(beta, X_hi, X_lo, comparison_type, family_type, link_type)
    return group_reducer(comp, groups, num_groups)


# Public comparison functions
comparison = _comparison
comparison_byT = _comparison_byT


# Public jacobian functions
jacobian = create_jacobian(_comparison)
jacobian_byT = create_jacobian_byT(_comparison_byT)
jacobian_byG = create_jacobian_byG(comparison_byG)
