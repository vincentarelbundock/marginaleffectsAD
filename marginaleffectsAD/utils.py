import jax
import jax.numpy as jnp
from jax import jit, jacrev
from jax import lax
import numpy as np


def array(X):
    return np.array(X)


@jit
def _standard_errors(J, V) -> jnp.ndarray:
    J = jnp.asarray(J)
    V = jnp.asarray(V)
    return jnp.sqrt(jnp.sum((J @ V) * J, axis=1))


def standard_errors(J, V) -> np.ndarray:
    se = _standard_errors(J, V)
    return np.array(se, dtype=np.float64)


def group_reducer(
    data: jnp.ndarray,
    groups: jnp.ndarray,
    num_groups: int,
) -> jnp.ndarray:
    """Group-based reducer function for averaging data by groups."""
    # Use stop_gradient to ensure num_groups remains concrete during differentiation
    num_groups_concrete = lax.stop_gradient(num_groups)
    group_sums = jax.ops.segment_sum(data, groups, num_segments=num_groups_concrete)
    group_counts = jnp.bincount(groups, length=num_groups_concrete)
    return group_sums / group_counts


def create_jacobian(func):
    """Factory function to create jacobian functions using automatic differentiation."""

    def jacobian(beta, *args, **kwargs) -> np.ndarray:
        return np.array(jacrev(lambda c: func(c, *args, **kwargs))(beta))

    return jacobian


# Aliases for clarity in different contexts
create_jacobian_byT = create_jacobian
create_jacobian_byG = create_jacobian
