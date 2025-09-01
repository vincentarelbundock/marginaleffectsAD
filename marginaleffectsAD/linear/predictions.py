import jax.numpy as jnp
import numpy as np
from jax import jit
from typing import Callable
from ..utils import group_reducer, create_jacobian_byG


def _predict_core(
    beta: jnp.ndarray,
    X: jnp.ndarray,
) -> jnp.ndarray:
    """Core prediction function for linear models - single source of truth."""
    return X @ beta


def _predict_reduced(
    beta: jnp.ndarray,
    X: jnp.ndarray,
    reducer: Callable[[jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    """Generic prediction with reducer function applied."""
    pred = _predict_core(beta, X)
    return reducer(pred)


@jit
def _predict(beta: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
    return _predict_reduced(beta, X, lambda x: x)


@jit
def _predict_byT(beta: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
    return _predict_reduced(beta, X, jnp.mean)


def predict_byG(
    beta: jnp.ndarray, X: jnp.ndarray, groups: jnp.ndarray, num_groups: int
) -> jnp.ndarray:
    pred = _predict_core(beta, X)
    return group_reducer(pred, groups, num_groups)


# Efficient jacobian functions (analytical solutions)
@jit
def _jacobian(beta: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
    return X


@jit
def _jacobian_byT(beta: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(X, axis=0)


# Public jacobian functions
def jacobian(beta: jnp.ndarray, X: jnp.ndarray, *args, **kwargs) -> np.ndarray:
    return np.array(_jacobian(beta, X))


def jacobian_byT(beta: jnp.ndarray, X: jnp.ndarray, *args, **kwargs) -> np.ndarray:
    return np.array(_jacobian_byT(beta, X))


# Public prediction functions
predict = _predict
predict_byT = _predict_byT


jacobian_byG = create_jacobian_byG(predict_byG)
