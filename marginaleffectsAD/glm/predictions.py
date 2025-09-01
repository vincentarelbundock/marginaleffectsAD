import jax
import jax.numpy as jnp
import numpy as np
from jax import jacfwd, jacrev, jit
from typing import Callable, Optional
from .families import Family, Link, linkinv, get_default_link
from ..utils import group_reducer, create_jacobian_byG


def _resolve_link(family_type: int, link_type: Optional[int]) -> int:
    """Resolve link type, using default if None."""
    return link_type if link_type is not None else get_default_link(family_type)


def _predict_core(
    beta: jnp.ndarray,
    X: jnp.ndarray,
    family_type: int,
    link_type: Optional[int],
) -> jnp.ndarray:
    """Core prediction function - single source of truth for prediction computation."""
    lt = _resolve_link(family_type, link_type)
    linear_pred = X @ beta
    return linkinv(lt, linear_pred)


@jit
def _predict(
    beta: jnp.ndarray, X: jnp.ndarray, family_type: int, link_type: int = None
) -> jnp.ndarray:
    return _predict_core(beta, X, family_type, link_type)


@jit
def _predict_byT(
    beta: jnp.ndarray, X: jnp.ndarray, family_type: int, link_type: int = None
) -> jnp.ndarray:
    pred = _predict_core(beta, X, family_type, link_type)
    return jnp.mean(pred)


@jit
def predict_byG(
    beta: jnp.ndarray,
    X: jnp.ndarray,
    groups: jnp.ndarray,
    num_groups: int,
    family_type: int,
    link_type: int = None,
) -> jnp.ndarray:
    pred = _predict_core(beta, X, family_type, link_type)
    return group_reducer(pred, groups, num_groups)


# Efficient jacobian functions
@jit
def _jacobian(
    beta: jnp.ndarray, X: jnp.ndarray, family_type: int, link_type: int = None
) -> jnp.ndarray:
    return jacfwd(_predict, argnums=0)(beta, X, family_type, link_type)


@jit
def _jacobian_byT(
    beta: jnp.ndarray, X: jnp.ndarray, family_type: int, link_type: int = None
) -> jnp.ndarray:
    return jacrev(_predict_byT, argnums=0)(beta, X, family_type, link_type)


# Public jacobian functions
def jacobian(
    beta: jnp.ndarray,
    X: jnp.ndarray,
    family_type: int,
    link_type: int = None,
    *args,
    **kwargs,
) -> np.ndarray:
    return np.array(_jacobian(beta, X, family_type, link_type))


def jacobian_byT(
    beta: jnp.ndarray,
    X: jnp.ndarray,
    family_type: int,
    link_type: int = None,
    *args,
    **kwargs,
) -> np.ndarray:
    return np.array(_jacobian_byT(beta, X, family_type, link_type))


jacobian_byG = create_jacobian_byG(predict_byG)


# Public prediction functions
predict = _predict
predict_byT = _predict_byT
