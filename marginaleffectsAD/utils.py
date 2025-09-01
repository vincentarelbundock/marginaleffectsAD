import jax
import jax.numpy as jnp
from jax import jit, jacrev
import numpy as np


def array(X):
    return np.array(X)


@jit
def standard_errors(J, V) -> jnp.ndarray:
    J = jnp.asarray(J)
    V = jnp.asarray(V)
    return jnp.sqrt(jnp.sum((J @ V) * J, axis=1))


def group_reducer(
    data: jnp.ndarray,
    groups: jnp.ndarray,
    num_groups: int,
) -> jnp.ndarray:
    """Group-based reducer function for averaging data by groups."""
    group_sums = jax.ops.segment_sum(data, groups, num_segments=num_groups)
    group_counts = jnp.bincount(groups, length=num_groups)
    return group_sums / group_counts


def create_jacobian(func):
    """Factory function to create jacobian functions."""

    def jacobian(beta, *args, **kwargs) -> np.ndarray:
        """Generic jacobian function."""
        return np.array(jacrev(lambda c: func(c, *args, **kwargs))(beta))

    return jacobian


def create_jacobian_byT(func):
    """Factory function to create jacobian_byT functions."""

    def jacobian_byT(beta, *args, **kwargs) -> np.ndarray:
        """Generic jacobian_byT function."""
        return np.array(jacrev(lambda c: func(c, *args, **kwargs))(beta))

    return jacobian_byT


def create_jacobian_byG(func):
    """Factory function to create jacobian_byG functions."""

    def jacobian_byG(beta, *args, **kwargs) -> np.ndarray:
        """Generic jacobian_byG function."""
        return np.array(jacrev(lambda c: func(c, *args, **kwargs))(beta))

    return jacobian_byG
