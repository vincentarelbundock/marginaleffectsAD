"""Comparison types and functions using enum-based approach for JAX compatibility."""

import jax.numpy as jnp
from jax import lax
from enum import IntEnum


class ComparisonType(IntEnum):
    """Comparison types for marginal effects."""

    DIFFERENCE = 0
    RATIO = 1
    LNRATIO = 2
    LNOR = 3
    LIFT = 4


def _compute_comparison(
    comparison_type: int, pred_hi: jnp.ndarray, pred_lo: jnp.ndarray
) -> jnp.ndarray:
    """Apply comparison function based on type (element-wise).

    Returns N-length array for element-wise comparisons.
    """
    return lax.switch(
        comparison_type,
        [
            lambda hi, lo: hi - lo,  # difference
            lambda hi, lo: hi / lo,  # ratio
            lambda hi, lo: jnp.log(hi / lo),  # lnratio
            lambda hi, lo: jnp.log((hi / (1 - hi)) / (lo / (1 - lo))),  # lnor
            lambda hi, lo: (hi - lo) / lo,  # lift
        ],
        pred_hi,
        pred_lo,
    )


def _compute_comparison_avg(
    comparison_type: int, pred_hi: jnp.ndarray, pred_lo: jnp.ndarray
) -> jnp.ndarray:
    """Apply comparison with averaging.

    For RATIO, LNRATIO, LNOR: aggregates first, then computes (mean(hi) / mean(lo))
    For others: computes element-wise, then aggregates (mean(hi - lo))

    Returns scalar.
    """
    # For ratio family: aggregate first, then compute
    if comparison_type in (
        ComparisonType.RATIO,
        ComparisonType.LNRATIO,
        ComparisonType.LNOR,
    ):
        hi_mean = jnp.mean(pred_hi)
        lo_mean = jnp.mean(pred_lo)
        return lax.switch(
            comparison_type,
            [
                lambda hm, lm: hm - lm,  # not used
                lambda hm, lm: hm / lm,  # ratioavg = mean(hi) / mean(lo)
                lambda hm, lm: jnp.log(
                    hm / lm
                ),  # lnratioavg = log(mean(hi) / mean(lo))
                lambda hm, lm: jnp.log((hm / (1 - hm)) / (lm / (1 - lm))),  # lnoravg
                lambda hm, lm: hm - lm,  # not used
            ],
            hi_mean,
            lo_mean,
        )
    else:
        # For others: compute element-wise, then aggregate
        comp = _compute_comparison(comparison_type, pred_hi, pred_lo)
        return jnp.mean(comp)


# Convenience instance for direct use
comparison_type = ComparisonType
