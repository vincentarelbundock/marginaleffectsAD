"""High-level functions that return estimates, Jacobians, and standard errors."""

import jax.numpy as jnp
import numpy as np
from typing import Optional


def predictions(
    beta: jnp.ndarray,
    X: jnp.ndarray,
    vcov: jnp.ndarray,
    family_type: Optional[int] = None,
    link_type: Optional[int] = None,
    by: bool = False,
    groups: Optional[jnp.ndarray] = None,
    num_groups: Optional[int] = None,
) -> dict:
    """Compute predictions with Jacobian and standard errors.

    Args:
        beta: Coefficient vector
        X: Design matrix
        vcov: Variance-covariance matrix
        family_type: GLM family type (from Family enum). If None, uses linear model.
        link_type: Link function type (from Link enum), optional (GLM only)
        by: If True, compute average predictions by group
        groups: Group indices (required if by=True). Must be sorted array of non-negative
            integers starting from 0.
        num_groups: Number of groups (optional, computed from groups if not provided)

    Returns:
        Dictionary with keys:
        - "estimate": Predictions (numpy array)
        - "jacobian": Jacobian matrix (numpy array)
        - "std_error": Standard errors (numpy array)
        - "groups": Group indices (numpy array, only present when by=True)
    """
    from .utils import standard_errors

    if by and groups is None:
        raise ValueError("groups required when by=True")

    # Compute num_groups if not provided
    if by:
        groups_array = np.asarray(groups)
        if num_groups is None:
            num_groups = int(np.max(groups_array) + 1)

    # Dispatch to GLM or linear model
    if family_type is not None:
        from .glm import predictions as glm_pred
        from .glm.families import get_default_link

        # Resolve link type to avoid None in JIT functions
        resolved_link = link_type if link_type is not None else get_default_link(family_type)

        if by:
            est = glm_pred.predict_byG(beta, X, groups, num_groups, family_type, resolved_link)
            jac = glm_pred.jacobian_byG(beta, X, groups, num_groups, family_type, resolved_link)
        else:
            est = glm_pred.predict(beta, X, family_type, resolved_link)
            jac = glm_pred.jacobian(beta, X, family_type, resolved_link)
    else:
        from .linear import predictions as lm_pred

        if by:
            est = lm_pred.predict_byG(beta, X, groups, num_groups)
            jac = lm_pred.jacobian_byG(beta, X, groups, num_groups)
        else:
            est = lm_pred.predict(beta, X)
            jac = lm_pred.jacobian(beta, X)

    se = standard_errors(jac, vcov)

    result = {
        "estimate": np.array(est),
        "jacobian": jac,
        "std_error": se,
    }

    if by:
        result["groups"] = np.arange(num_groups)

    return result


def comparisons(
    beta: jnp.ndarray,
    X_hi: jnp.ndarray,
    X_lo: jnp.ndarray,
    vcov: jnp.ndarray,
    comparison_type: int,
    family_type: Optional[int] = None,
    link_type: Optional[int] = None,
    by: bool = False,
    groups: Optional[jnp.ndarray] = None,
    num_groups: Optional[int] = None,
) -> dict:
    """Compute comparisons with Jacobian and standard errors.

    Args:
        beta: Coefficient vector
        X_hi: Design matrix for high values
        X_lo: Design matrix for low values
        vcov: Variance-covariance matrix
        comparison_type: Type of comparison (from ComparisonType enum)
        family_type: GLM family type (from Family enum). If None, uses linear model.
        link_type: Link function type (from Link enum), optional (GLM only)
        by: If True, compute average comparisons by group
        groups: Group indices (required if by=True). Must be sorted array of non-negative
            integers starting from 0.
        num_groups: Number of groups (optional, computed from groups if not provided)

    Returns:
        Dictionary with keys:
        - "estimate": Comparisons (numpy array)
        - "jacobian": Jacobian matrix (numpy array)
        - "std_error": Standard errors (numpy array)
        - "groups": Group indices (numpy array, only present when by=True)
    """
    from .utils import standard_errors

    if by and groups is None:
        raise ValueError("groups required when by=True")

    # Compute num_groups if not provided
    if by:
        groups_array = np.asarray(groups)
        if num_groups is None:
            num_groups = int(np.max(groups_array) + 1)

    # Dispatch to GLM or linear model
    if family_type is not None:
        from .glm import comparisons as glm_comp
        from .glm.families import get_default_link

        # Resolve link type to avoid None in JIT functions
        resolved_link = link_type if link_type is not None else get_default_link(family_type)

        if by:
            est = glm_comp.comparison_byG(
                beta, X_hi, X_lo, groups, num_groups, comparison_type, family_type, resolved_link
            )
            jac = glm_comp.jacobian_byG(
                beta, X_hi, X_lo, groups, num_groups, comparison_type, family_type, resolved_link
            )
        else:
            est = glm_comp.comparison(beta, X_hi, X_lo, comparison_type, family_type, resolved_link)
            jac = glm_comp.jacobian(beta, X_hi, X_lo, comparison_type, family_type, resolved_link)
    else:
        from .linear import comparisons as lm_comp

        if by:
            est = lm_comp.comparison_byG(beta, X_hi, X_lo, groups, num_groups, comparison_type)
            jac = lm_comp.jacobian_byG(beta, X_hi, X_lo, groups, num_groups, comparison_type)
        else:
            est = lm_comp.comparison(beta, X_hi, X_lo, comparison_type)
            jac = lm_comp.jacobian(beta, X_hi, X_lo, comparison_type)

    se = standard_errors(jac, vcov)

    result = {
        "estimate": np.array(est),
        "jacobian": jac,
        "std_error": se,
    }

    if by:
        result["groups"] = np.arange(num_groups)

    return result
