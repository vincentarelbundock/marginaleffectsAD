import jax
import jax.numpy as jnp
import numpy as np
from jax import jacrev, jit


def _difference(beta: jnp.ndarray, X_hi: jnp.ndarray, X_lo: jnp.ndarray) -> jnp.ndarray:
    pred_hi = X_hi @ beta
    pred_lo = X_lo @ beta
    return pred_hi - pred_lo


def _difference_byT(
    beta: jnp.ndarray, X_hi: jnp.ndarray, X_lo: jnp.ndarray
) -> jnp.ndarray:
    pred_hi = X_hi @ beta
    pred_lo = X_lo @ beta
    comp = pred_hi - pred_lo
    return jnp.mean(comp)


def difference_byG(
    beta: jnp.ndarray,
    X_hi: jnp.ndarray,
    X_lo: jnp.ndarray,
    groups: jnp.ndarray,
    num_groups: int,
) -> jnp.ndarray:
    pred_hi = X_hi @ beta
    pred_lo = X_lo @ beta
    comp = pred_hi - pred_lo
    group_sums = jax.ops.segment_sum(comp, groups, num_segments=num_groups)
    group_counts = jnp.bincount(groups, length=num_groups)
    return group_sums / group_counts


def _ratio(beta: jnp.ndarray, X_hi: jnp.ndarray, X_lo: jnp.ndarray) -> jnp.ndarray:
    pred_hi = X_hi @ beta
    pred_lo = X_lo @ beta
    return pred_hi / pred_lo


def _ratio_byT(beta: jnp.ndarray, X_hi: jnp.ndarray, X_lo: jnp.ndarray) -> jnp.ndarray:
    pred_hi = X_hi @ beta
    pred_lo = X_lo @ beta
    ratio_vals = pred_hi / pred_lo
    return jnp.mean(ratio_vals)


def ratio_byG(
    beta: jnp.ndarray,
    X_hi: jnp.ndarray,
    X_lo: jnp.ndarray,
    groups: jnp.ndarray,
    num_groups: int,
) -> jnp.ndarray:
    pred_hi = X_hi @ beta
    pred_lo = X_lo @ beta
    ratio_vals = pred_hi / pred_lo
    group_sums = jax.ops.segment_sum(ratio_vals, groups, num_segments=num_groups)
    group_counts = jnp.bincount(groups, length=num_groups)
    return group_sums / group_counts


def _jacobian_difference(
    beta: jnp.ndarray, X_hi: jnp.ndarray, X_lo: jnp.ndarray
) -> jnp.ndarray:
    return X_hi - X_lo


def _jacobian_difference_byT(
    beta: jnp.ndarray, X_hi: jnp.ndarray, X_lo: jnp.ndarray
) -> jnp.ndarray:
    return jnp.mean(X_hi - X_lo, axis=0)


def jacobian_difference_byG(
    beta: jnp.ndarray,
    X_hi: jnp.ndarray,
    X_lo: jnp.ndarray,
    groups: jnp.ndarray,
    num_groups: int,
) -> np.ndarray:
    return np.array(
        jacrev(lambda c: difference_byG(c, X_hi, X_lo, groups, num_groups))(beta)
    )


def _jacobian_ratio(
    beta: jnp.ndarray, X_hi: jnp.ndarray, X_lo: jnp.ndarray
) -> jnp.ndarray:
    return jacrev(lambda c: _ratio(c, X_hi, X_lo))(beta)


def _jacobian_ratio_byT(
    beta: jnp.ndarray, X_hi: jnp.ndarray, X_lo: jnp.ndarray
) -> jnp.ndarray:
    return jacrev(lambda c: _ratio_byT(c, X_hi, X_lo))(beta)


def jacobian_ratio_byG(
    beta: jnp.ndarray,
    X_hi: jnp.ndarray,
    X_lo: jnp.ndarray,
    groups: jnp.ndarray,
    num_groups: int,
) -> np.ndarray:
    return np.array(
        jacrev(lambda c: ratio_byG(c, X_hi, X_lo, groups, num_groups))(beta)
    )


# JIT compiled versions of internal functions
difference = jit(_difference)
difference_byT = jit(_difference_byT)
ratio = jit(_ratio)
ratio_byT = jit(_ratio_byT)
_jacobian_difference_jit = jit(_jacobian_difference)
_jacobian_difference_byT_jit = jit(_jacobian_difference_byT)
_jacobian_ratio_jit = jit(_jacobian_ratio)
_jacobian_ratio_byT_jit = jit(_jacobian_ratio_byT)


# Public jacobian functions that return numpy arrays
def jacobian_difference(
    beta: jnp.ndarray, X_hi: jnp.ndarray, X_lo: jnp.ndarray, *args, **kwargs
) -> np.ndarray:
    return np.array(_jacobian_difference_jit(beta, X_hi, X_lo))


def jacobian_difference_byT(
    beta: jnp.ndarray, X_hi: jnp.ndarray, X_lo: jnp.ndarray, *args, **kwargs
) -> np.ndarray:
    return np.array(_jacobian_difference_byT_jit(beta, X_hi, X_lo))


def jacobian_ratio(
    beta: jnp.ndarray, X_hi: jnp.ndarray, X_lo: jnp.ndarray, *args, **kwargs
) -> np.ndarray:
    return np.array(_jacobian_ratio_jit(beta, X_hi, X_lo))


def jacobian_ratio_byT(
    beta: jnp.ndarray, X_hi: jnp.ndarray, X_lo: jnp.ndarray, *args, **kwargs
) -> np.ndarray:
    return np.array(_jacobian_ratio_byT_jit(beta, X_hi, X_lo))


# Note: *_byG functions cannot be JIT compiled due to num_groups parameter
