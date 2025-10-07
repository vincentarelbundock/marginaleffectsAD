import numpy as np
import polars as pl
from marginaleffects import get_dataset
from formulaic import model_matrix
import statsmodels.api as sm
from marginaleffectsAD.glm import predictions as glm_predictions
from marginaleffectsAD.glm.families import Family, Link
from marginaleffectsAD.utils import standard_errors

# Step 1: Download mtcars dataset
mtcars = get_dataset("mtcars", "datasets").to_pandas()
y, X = model_matrix("am ~ hp * wt + C(cyl)", mtcars)
y = y.values
X = X.values

# Step 2: Fit logit model using statsmodels
logit_result = sm.Logit(y, X).fit()

# Step 3: Extract coefficients and variance-covariance matrix
beta = logit_result.params
V = logit_result.cov_params()

print("=" * 60)
print("EXAMPLE 1: Low-level API (manual workflow)")
print("=" * 60)

# Step 4: Make predictions using JAX implementation
jax_predictions = glm_predictions.predict(beta, X, Family.BINOMIAL, Link.LOGIT)

# Step 5: Compute standard errors using JAX
J = glm_predictions.jacobian(beta, X, Family.BINOMIAL, Link.LOGIT)
jax_standard_errors = standard_errors(J, V)

results = pl.DataFrame(
    {"estimate": np.array(jax_predictions), "std_error": np.array(jax_standard_errors)}
)
print(results)

print("\n" + "=" * 60)
print("EXAMPLE 2: High-level predictions() function")
print("=" * 60)

from marginaleffectsAD import predictions, comparisons, Family, Link, ComparisonType

# Single function call returns estimate, jacobian, and standard errors
result = predictions(beta, X, V, family_type=Family.BINOMIAL, link_type=Link.LOGIT)

results_hl = pl.DataFrame(
    {"estimate": result["estimate"], "std_error": result["std_error"]}
)
print(results_hl)
print(f"\nJacobian shape: {result['jacobian'].shape}")


# Fit linear model
lm_result = sm.OLS(mtcars["mpg"], X).fit()
beta_lm = np.array(lm_result.params)
V_lm = np.array(lm_result.cov_params())

# Use predictions() without family_type for linear models
result_lm = predictions(beta_lm, X, V_lm)

print(pl.DataFrame({
    "estimate": result_lm["estimate"][:5],
    "std_error": result_lm["std_error"][:5]
}))

print("\n" + "=" * 60)
print("EXAMPLE 4: Comparisons (contrasts)")
print("=" * 60)

# Create counterfactual datasets: +1 unit change in hp (column index 1)
X_hi = X.copy()
X_lo = X.copy()
X_hi[:, 1] = X[:, 1] + 1  # hp + 1
X_lo[:, 1] = X[:, 1]      # hp (original)

# Compute difference in predictions
result_comp = comparisons(
    beta, X_hi, X_lo, V,
    comparison_type=ComparisonType.DIFFERENCE,
    family_type=Family.BINOMIAL,
    link_type=Link.LOGIT
)

print(pl.DataFrame({
    "estimate": result_comp["estimate"][:5],
    "std_error": result_comp["std_error"][:5]
}))
print(f"\nMean effect: {result_comp['estimate'].mean():.4f}")

print("\n" + "=" * 60)
print("EXAMPLE 5: Ratio comparison")
print("=" * 60)

# Compute ratio instead of difference
result_ratio = comparisons(
    beta, X_hi, X_lo, V,
    comparison_type=ComparisonType.RATIO,
    family_type=Family.BINOMIAL,
    link_type=Link.LOGIT
)

print(pl.DataFrame({
    "estimate": result_ratio["estimate"][:5],
    "std_error": result_ratio["std_error"][:5]
}))
print(f"\nMean ratio: {result_ratio['estimate'].mean():.4f}")

print("\n" + "=" * 60)
print("EXAMPLE 6: Group aggregation")
print("=" * 60)

# Create groups based on cylinder count
cyl_groups = mtcars["cyl"].values
unique_cyls = np.unique(cyl_groups)
group_map = {cyl: i for i, cyl in enumerate(unique_cyls)}
groups = np.array([group_map[cyl] for cyl in cyl_groups])
num_groups = len(unique_cyls)

# Compute predictions by group
result_byG = predictions(
    beta, X, V,
    family_type=Family.BINOMIAL,
    link_type=Link.LOGIT,
    by_group=True,
    groups=groups,
    num_groups=num_groups
)

print(pl.DataFrame({
    "cylinder": unique_cyls,
    "group_id": result_byG["groups"],
    "estimate": result_byG["estimate"],
    "std_error": result_byG["std_error"]
}))

# Compute comparisons by group
result_comp_byG = comparisons(
    beta, X_hi, X_lo, V,
    comparison_type=ComparisonType.DIFFERENCE,
    family_type=Family.BINOMIAL,
    link_type=Link.LOGIT,
    by_group=True,
    groups=groups,
    num_groups=num_groups
)

print("\nComparisons by cylinder group:")
print(pl.DataFrame({
    "cylinder": unique_cyls,
    "group_id": result_comp_byG["groups"],
    "estimate": result_comp_byG["estimate"],
    "std_error": result_comp_byG["std_error"]
}))
