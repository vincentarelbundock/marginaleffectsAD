# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`marginaleffectsAD` is a Python package that provides JAX-based automatic differentiation helpers for computing marginal effects in statistical models. The package uses JAX for efficient gradient computation and supports both linear models and generalized linear models (GLMs).

## Build and Development Commands

This project uses `uv` for package management. All commands are defined in the Makefile:

- **Install package**: `make install` or `uv pip install .`
- **Run linter**: `make lint` (runs ruff check and format)
- **Interactive Python shell**: `make ipy` or `uv run --all-extras ipython --no-autoindent`

Note: There is no test suite currently in this repository (the `make test` command exists but there are no test files).

## Architecture

The package is organized into three main modules:

### 1. Core Utilities (`marginaleffectsAD/utils.py`)

Provides foundational JAX functions:
- `standard_errors(J, V)`: Computes standard errors from Jacobian matrix and variance-covariance matrix
- `group_reducer()`: Averages predictions/comparisons by group using JAX segment operations
- Factory functions `create_jacobian()`, `create_jacobian_byT()`, `create_jacobian_byG()`: Generate Jacobian computation functions via `jacrev`

### 2. Linear Models (`marginaleffectsAD/linear/`)

Implements predictions and comparisons for linear models:
- **Predictions** (`predictions.py`): Uses analytical Jacobians (since ∂(Xβ)/∂β = X)
  - `predict()`: Row-level predictions
  - `predict_byT()`: Average prediction across all rows
  - `predict_byG()`: Average prediction by group
- **Comparisons** (`comparisons.py`): Computes differences/ratios between high/low predictions
  - Uses factory functions from utils to create Jacobians via automatic differentiation

### 3. GLM Models (`marginaleffectsAD/glm/`)

Implements predictions and comparisons for generalized linear models:
- **Families** (`families.py`): Defines GLM families (Gaussian, Binomial, Poisson, Gamma, Inverse Gaussian) and link functions (identity, log, logit, probit, inverse, sqrt, cloglog) using `IntEnum` for JAX compatibility
  - `linkinv()`: Inverse link function (η → μ)
  - `linkfun()`: Link function (μ → η)
  - Uses `lax.switch()` for efficient dispatch based on family/link type
- **Predictions** (`predictions.py`): Computes predictions via X @ β → linkinv()
  - Uses `jacfwd` for forward-mode AD (efficient for large datasets)
  - Supports same aggregation modes as linear models (row-level, byT, byG)
- **Comparisons** (`comparisons.py`): Similar structure to linear comparisons but applies link functions

### 4. Comparison Types (`marginaleffectsAD/comparisons.py`)

Defines comparison operations using `IntEnum`:
- `ComparisonType.DIFFERENCE`: hi - lo
- `ComparisonType.RATIO`: hi / lo
- Uses `lax.switch()` for JAX-compatible dispatch

## Design Patterns

**Core Functions Pattern**: Each module uses a `_core` function as the single source of truth, with specialized wrappers for different aggregation modes:
- `_predict_core()` / `_comparison_core()`: Core computation logic
- Public functions (`predict`, `predict_byT`, `predict_byG`) call the core function with different reducers

**JIT Compilation**: Most functions decorated with `@jit` for performance, except `*_byG` functions (cannot JIT due to dynamic `num_groups` parameter)

**Return Types**: Internal functions use JAX arrays (`jnp.ndarray`), public APIs return NumPy arrays (`np.ndarray`)

**JAX Compatibility**: Uses `IntEnum` and `lax.switch()` instead of string-based dispatch or dictionaries to ensure compatibility with JAX's tracing and compilation

## Example Usage

See `example.py` for a complete workflow:
1. Fit a GLM model using statsmodels
2. Use `marginaleffectsAD` to compute predictions with `logit.predict(beta, X)`
3. Compute Jacobian with `logit.jacobian(beta, X)`
4. Calculate standard errors using `standard_errors(J, V)`

## Dependencies

- **Core**: `jax`, `numpy`
- **Dev**: `formulaic`, `marginaleffects`, `polars`, `statsmodels`, `ruff`
- Requires Python 3.11+
