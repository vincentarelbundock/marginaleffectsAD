"""Comparison types and functions using enum-based approach for JAX compatibility."""

import jax.numpy as jnp
from jax import lax
from enum import IntEnum


class ComparisonType(IntEnum):
    """Comparison types for marginal effects."""

    DIFFERENCE = 0
    RATIO = 1


def _compute_comparison(
    comparison_type: int, pred_hi: jnp.ndarray, pred_lo: jnp.ndarray
) -> jnp.ndarray:
    """Apply comparison function based on type."""
    return lax.switch(
        comparison_type,
        [
            lambda hi, lo: hi - lo,  # difference
            lambda hi, lo: hi / lo,  # ratio
        ],
        pred_hi,
        pred_lo,
    )


# Convenience instance for direct use
comparison_type = ComparisonType
