import jax
import jax.numpy as jnp
import numpy as np
from jax import jacfwd, jacrev, jit
from .families import Family, Link, linkinv, get_default_link


def _predict(
    beta: jnp.ndarray, X: jnp.ndarray, family_type: int, link_type: int = None
) -> jnp.ndarray:
    if link_type is None:
        link_type = get_default_link(family_type)
    linear_pred = X @ beta
    return linkinv(link_type, linear_pred)


def _predict_byT(
    beta: jnp.ndarray, X: jnp.ndarray, family_type: int, link_type: int = None
) -> jnp.ndarray:
    return jnp.mean(_predict(beta, X, family_type, link_type))


def predict_byG(
    beta: jnp.ndarray,
    X: jnp.ndarray,
    groups: jnp.ndarray,
    num_groups: int,
    family_type: int,
    link_type: int = None,
) -> jnp.ndarray:
    preds = _predict(beta, X, family_type, link_type)
    group_sums = jax.ops.segment_sum(preds, groups, num_segments=num_groups)
    group_counts = jnp.bincount(groups, length=num_groups)
    return group_sums / group_counts


def _jacobian(
    beta: jnp.ndarray, X: jnp.ndarray, family_type: int, link_type: int = None
) -> jnp.ndarray:
    return jacfwd(lambda c: _predict(c, X, family_type, link_type))(beta)


def _jacobian_byT(
    beta: jnp.ndarray, X: jnp.ndarray, family_type: int, link_type: int = None
) -> jnp.ndarray:
    return jacrev(lambda c: _predict_byT(c, X, family_type, link_type))(beta)


def jacobian_byG(
    beta: jnp.ndarray,
    X: jnp.ndarray,
    groups: jnp.ndarray,
    num_groups: int,
    family_type: int,
    link_type: int = None,
    *args,
    **kwargs,
) -> np.ndarray:
    return np.array(
        jacrev(lambda c: predict_byG(c, X, groups, num_groups, family_type, link_type))(
            beta
        )
    )


# JIT compiled versions of internal functions
predict = jit(_predict)
predict_byT = jit(_predict_byT)
_jacobian_jit = jit(_jacobian)
_jacobian_byT_jit = jit(_jacobian_byT)


# Public jacobian functions that return numpy arrays
def jacobian(
    beta: jnp.ndarray,
    X: jnp.ndarray,
    family_type: int,
    link_type: int = None,
    *args,
    **kwargs,
) -> np.ndarray:
    return np.array(_jacobian_jit(beta, X, family_type, link_type))


def jacobian_byT(
    beta: jnp.ndarray,
    X: jnp.ndarray,
    family_type: int,
    link_type: int = None,
    *args,
    **kwargs,
) -> np.ndarray:
    return np.array(_jacobian_byT_jit(beta, X, family_type, link_type))


# Convenience functions for common cases (backward compatibility)
def predict_gaussian(
    beta: jnp.ndarray, X: jnp.ndarray, link_type: int = Link.IDENTITY
) -> jnp.ndarray:
    return predict(beta, X, Family.GAUSSIAN, link_type)


def predict_binomial(
    beta: jnp.ndarray, X: jnp.ndarray, link_type: int = Link.LOGIT
) -> jnp.ndarray:
    return predict(beta, X, Family.BINOMIAL, link_type)


def predict_poisson(
    beta: jnp.ndarray, X: jnp.ndarray, link_type: int = Link.LOG
) -> jnp.ndarray:
    return predict(beta, X, Family.POISSON, link_type)


def predict_gamma(
    beta: jnp.ndarray, X: jnp.ndarray, link_type: int = Link.INVERSE
) -> jnp.ndarray:
    return predict(beta, X, Family.GAMMA, link_type)

