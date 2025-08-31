import jax
import jax.numpy as jnp
import numpy as np
from jax import jacrev, jit


def _predict(beta: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
    return X @ beta


def _predict_byT(beta: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(X @ beta)


def predict_byG(
    beta: jnp.ndarray, X: jnp.ndarray, groups: jnp.ndarray, num_groups: int
) -> jnp.ndarray:
    preds = X @ beta
    group_sums = jax.ops.segment_sum(preds, groups, num_segments=num_groups)
    group_counts = jnp.bincount(groups, length=num_groups)
    return group_sums / group_counts


def _jacobian(beta: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
    return X


def _jacobian_byT(beta: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(X, axis=0)


def jacobian_byG(
    beta: jnp.ndarray,
    X: jnp.ndarray,
    groups: jnp.ndarray,
    num_groups: int,
    *args,
    **kwargs,
) -> np.ndarray:
    return np.array(jacrev(lambda c: predict_byG(c, X, groups, num_groups))(beta))


# JIT compiled versions of internal functions
predict = jit(_predict)
predict_byT = jit(_predict_byT)
_jacobian_jit = jit(_jacobian)
_jacobian_byT_jit = jit(_jacobian_byT)


# Public jacobian functions that return numpy arrays
def jacobian(beta: jnp.ndarray, X: jnp.ndarray, *args, **kwargs) -> np.ndarray:
    return np.array(_jacobian_jit(beta, X))


def jacobian_byT(beta: jnp.ndarray, X: jnp.ndarray, *args, **kwargs) -> np.ndarray:
    return np.array(_jacobian_byT_jit(beta, X))


# Note: predict_byG and jacobian_byG cannot be JIT compiled due to num_groups parameter
