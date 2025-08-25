import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev, jit


def _predict(coefs: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.sigmoid(X @ coefs)


def _predict_byT(coefs: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jax.nn.sigmoid(X @ coefs))


def predict_byG(
    coefs: jnp.ndarray, X: jnp.ndarray, groups: jnp.ndarray, num_groups: int
) -> jnp.ndarray:
    preds = jax.nn.sigmoid(X @ coefs)
    group_sums = jax.ops.segment_sum(preds, groups, num_segments=num_groups)
    group_counts = jnp.bincount(groups, length=num_groups)
    return group_sums / group_counts


def _jacobian(coefs: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
    return jacfwd(lambda c: predict(c, X))(coefs)


def _jacobian_byT(coefs: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
    return jacrev(lambda c: predict_byT(c, X))(coefs)


def jacobian_byG(
    coefs: jnp.ndarray, X: jnp.ndarray, groups: jnp.ndarray, num_groups: int
) -> jnp.ndarray:
    return jacrev(lambda c: predict_byG(c, X, groups, num_groups))(coefs)


predict = jit(_predict)
predict_byT = jit(_predict_byT)
jacobian = jit(_jacobian)
jacobian_byT = jit(_jacobian_byT)
# Note: predict_byG and jacobian_byG cannot be JIT compiled due to num_groups parameter
