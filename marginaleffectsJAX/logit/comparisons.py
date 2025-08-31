import jax
import jax.numpy as jnp
import numpy as np
from jax import jacrev, jit
from .predictions import predict, predict_byT, predict_byG


def _difference(beta: jnp.ndarray, X_hi: jnp.ndarray, X_lo: jnp.ndarray) -> jnp.ndarray:
    pred_hi = jax.nn.sigmoid(X_hi @ beta)
    pred_lo = jax.nn.sigmoid(X_lo @ beta)
    return pred_hi - pred_lo


def _difference_byT(
    beta: jnp.ndarray, X_hi: jnp.ndarray, X_lo: jnp.ndarray
) -> jnp.ndarray:
    pred_hi = jax.nn.sigmoid(X_hi @ beta)
    pred_lo = jax.nn.sigmoid(X_lo @ beta)
    comp = pred_hi - pred_lo
    return jnp.mean(comp)


def difference_byG(
    beta: jnp.ndarray,
    X_hi: jnp.ndarray,
    X_lo: jnp.ndarray,
    groups: jnp.ndarray,
    num_groups: int,
) -> jnp.ndarray:
    pred_hi = jax.nn.sigmoid(X_hi @ beta)
    pred_lo = jax.nn.sigmoid(X_lo @ beta)
    comp = pred_hi - pred_lo
    group_sums = jax.ops.segment_sum(comp, groups, num_segments=num_groups)
    group_counts = jnp.bincount(groups, length=num_groups)
    return group_sums / group_counts


def _ratio(beta: jnp.ndarray, X_hi: jnp.ndarray, X_lo: jnp.ndarray) -> jnp.ndarray:
    pred_hi = jax.nn.sigmoid(X_hi @ beta)
    pred_lo = jax.nn.sigmoid(X_lo @ beta)
    return pred_hi / pred_lo


def _ratio_byT(beta: jnp.ndarray, X_hi: jnp.ndarray, X_lo: jnp.ndarray) -> jnp.ndarray:
    pred_hi = jax.nn.sigmoid(X_hi @ beta)
    pred_lo = jax.nn.sigmoid(X_lo @ beta)
    ratio_vals = pred_hi / pred_lo
    return jnp.mean(ratio_vals)


def ratio_byG(
    beta: jnp.ndarray,
    X_hi: jnp.ndarray,
    X_lo: jnp.ndarray,
    groups: jnp.ndarray,
    num_groups: int,
) -> jnp.ndarray:
    pred_hi = jax.nn.sigmoid(X_hi @ beta)
    pred_lo = jax.nn.sigmoid(X_lo @ beta)
    ratio_vals = pred_hi / pred_lo
    group_sums = jax.ops.segment_sum(ratio_vals, groups, num_segments=num_groups)
    group_counts = jnp.bincount(groups, length=num_groups)
    return group_sums / group_counts


def _difference_jacobian(
    beta: jnp.ndarray, X_hi: jnp.ndarray, X_lo: jnp.ndarray
) -> jnp.ndarray:
    return jacrev(lambda c: _difference(c, X_hi, X_lo))(beta)


def _difference_jacobian_byT(
    beta: jnp.ndarray, X_hi: jnp.ndarray, X_lo: jnp.ndarray
) -> jnp.ndarray:
    return jacrev(lambda c: _difference_byT(c, X_hi, X_lo))(beta)


def difference_jacobian_byG(
    beta: jnp.ndarray,
    X_hi: jnp.ndarray,
    X_lo: jnp.ndarray,
    groups: jnp.ndarray,
    num_groups: int,
    *args, **kwargs
) -> np.ndarray:
    return np.array(
        jacrev(lambda c: difference_byG(c, X_hi, X_lo, groups, num_groups))(beta)
    )


def _ratio_jacobian(
    beta: jnp.ndarray, X_hi: jnp.ndarray, X_lo: jnp.ndarray
) -> jnp.ndarray:
    return jacrev(lambda c: _ratio(c, X_hi, X_lo))(beta)


def _ratio_jacobian_byT(
    beta: jnp.ndarray, X_hi: jnp.ndarray, X_lo: jnp.ndarray
) -> jnp.ndarray:
    return jacrev(lambda c: _ratio_byT(c, X_hi, X_lo))(beta)


def ratio_jacobian_byG(
    beta: jnp.ndarray,
    X_hi: jnp.ndarray,
    X_lo: jnp.ndarray,
    groups: jnp.ndarray,
    num_groups: int,
    *args, **kwargs
) -> np.ndarray:
    return np.array(
        jacrev(lambda c: ratio_byG(c, X_hi, X_lo, groups, num_groups))(beta)
    )


# JIT compiled versions of internal functions
difference = jit(_difference)
difference_byT = jit(_difference_byT)
ratio = jit(_ratio)
ratio_byT = jit(_ratio_byT)
_difference_jacobian_jit = jit(_difference_jacobian)
_difference_jacobian_byT_jit = jit(_difference_jacobian_byT)
_ratio_jacobian_jit = jit(_ratio_jacobian)
_ratio_jacobian_byT_jit = jit(_ratio_jacobian_byT)


# Public jacobian functions that return numpy arrays
def difference_jacobian(
    beta: jnp.ndarray, X_hi: jnp.ndarray, X_lo: jnp.ndarray, *args, **kwargs
) -> np.ndarray:
    return np.array(_difference_jacobian_jit(beta, X_hi, X_lo))


def difference_jacobian_byT(
    beta: jnp.ndarray, X_hi: jnp.ndarray, X_lo: jnp.ndarray, *args, **kwargs
) -> np.ndarray:
    return np.array(_difference_jacobian_byT_jit(beta, X_hi, X_lo))


def ratio_jacobian(
    beta: jnp.ndarray, X_hi: jnp.ndarray, X_lo: jnp.ndarray, *args, **kwargs
) -> np.ndarray:
    return np.array(_ratio_jacobian_jit(beta, X_hi, X_lo))


def ratio_jacobian_byT(
    beta: jnp.ndarray, X_hi: jnp.ndarray, X_lo: jnp.ndarray, *args, **kwargs
) -> np.ndarray:
    return np.array(_ratio_jacobian_byT_jit(beta, X_hi, X_lo))


# Note: *_byG functions cannot be JIT compiled due to num_groups parameter

