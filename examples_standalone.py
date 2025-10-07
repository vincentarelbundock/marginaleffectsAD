import numpy as np
import polars as pl
from marginaleffects import get_dataset
from formulaic import model_matrix
import statsmodels.api as sm
from marginaleffectsAD.linear.predictions import predictions, predictions_byT, predictions_byG
from marginaleffectsAD.linear.comparisons import comparisons, comparisons_byT, comparisons_byG
from marginaleffectsAD.glm import predictions as glm_predictions
from marginaleffectsAD.glm import comparisons as glm_comparisons
from marginaleffectsAD.glm.families import Family, Link
from marginaleffectsAD.comparisons import ComparisonType

# Download and prepare data
mtcars = get_dataset("mtcars", "datasets").to_pandas()
y, X = model_matrix("mpg ~ hp * wt + C(cyl)", mtcars)
y = y.values
X = X.values

# Fit linear model
lm_result = sm.OLS(y, X).fit()
beta = np.array(lm_result.params)
vcov = np.array(lm_result.cov_params())

print("=" * 60)
print("Unit-level predictions with predictions()")
print("=" * 60)

# Compute unit-level predictions, jacobian, and standard errors
result = predictions(beta, X, vcov)

print(f"Estimate shape: {result['estimate'].shape}")
print(f"Jacobian shape: {result['jacobian'].shape}")
print(f"Std error shape: {result['std_error'].shape}")
print()

# Display first 5 predictions
results_df = pl.DataFrame({
    "estimate": result["estimate"][:5],
    "std_error": result["std_error"][:5]
})
print(results_df)

print("\n" + "=" * 60)
print("Mean prediction with predictions_byT()")
print("=" * 60)

# Compute mean prediction across all units
result_byT = predictions_byT(beta, X, vcov)

print(f"Mean estimate: {result_byT['estimate']}")
print(f"Standard error: {result_byT['std_error']}")
print(f"Jacobian shape: {result_byT['jacobian'].shape}")
print()

# Verify it matches the mean of unit-level predictions
print(f"Verification - mean of unit predictions: {result['estimate'].mean():.6f}")
print(f"Verification - byT prediction:           {result_byT['estimate']:.6f}")

print("\n" + "=" * 60)
print("Group-level predictions with predictions_byG()")
print("=" * 60)

# Create groups based on cylinder count
cyl_groups = mtcars["cyl"].values
unique_cyls = np.unique(cyl_groups)
group_map = {cyl: i for i, cyl in enumerate(unique_cyls)}
groups = np.array([group_map[cyl] for cyl in cyl_groups])
num_groups = len(unique_cyls)

# Compute predictions by group
result_byG = predictions_byG(beta, X, vcov, groups, num_groups)

print(f"Estimate shape: {result_byG['estimate'].shape}")
print(f"Jacobian shape: {result_byG['jacobian'].shape}")
print(f"Std error shape: {result_byG['std_error'].shape}")
print()

results_byG_df = pl.DataFrame({
    "cylinder": unique_cyls,
    "estimate": result_byG["estimate"],
    "std_error": result_byG["std_error"]
})
print(results_byG_df)

print("\n" + "=" * 60)
print("GLM: Unit-level predictions")
print("=" * 60)

# Prepare data for logit model
y_logit, X_logit = model_matrix("am ~ hp * wt + C(cyl)", mtcars)
y_logit = y_logit.values
X_logit = X_logit.values

# Fit logit model
logit_result = sm.Logit(y_logit, X_logit).fit()
beta_logit = np.array(logit_result.params)
vcov_logit = np.array(logit_result.cov_params())

# Compute unit-level predictions
result_glm = glm_predictions.predictions(beta_logit, X_logit, vcov_logit, Family.BINOMIAL, Link.LOGIT)

print(f"Estimate shape: {result_glm['estimate'].shape}")
print(f"Jacobian shape: {result_glm['jacobian'].shape}")
print(f"Std error shape: {result_glm['std_error'].shape}")
print()

results_glm_df = pl.DataFrame({
    "estimate": result_glm["estimate"][:5],
    "std_error": result_glm["std_error"][:5]
})
print(results_glm_df)

print("\n" + "=" * 60)
print("GLM: Mean prediction")
print("=" * 60)

result_glm_byT = glm_predictions.predictions_byT(beta_logit, X_logit, vcov_logit, Family.BINOMIAL, Link.LOGIT)

print(f"Mean estimate: {result_glm_byT['estimate']}")
print(f"Standard error: {result_glm_byT['std_error']}")
print(f"Jacobian shape: {result_glm_byT['jacobian'].shape}")
print()
print(f"Verification - mean of unit predictions: {result_glm['estimate'].mean():.6f}")
print(f"Verification - byT prediction:           {result_glm_byT['estimate']:.6f}")

print("\n" + "=" * 60)
print("GLM: Group-level predictions")
print("=" * 60)

result_glm_byG = glm_predictions.predictions_byG(
    beta_logit, X_logit, vcov_logit, groups, num_groups, Family.BINOMIAL, Link.LOGIT
)

print(f"Estimate shape: {result_glm_byG['estimate'].shape}")
print(f"Jacobian shape: {result_glm_byG['jacobian'].shape}")
print(f"Std error shape: {result_glm_byG['std_error'].shape}")
print()

results_glm_byG_df = pl.DataFrame({
    "cylinder": unique_cyls,
    "estimate": result_glm_byG["estimate"],
    "std_error": result_glm_byG["std_error"]
})
print(results_glm_byG_df)

print("\n" + "=" * 60)
print("Linear: Unit-level comparisons")
print("=" * 60)

# Create counterfactual datasets: +1 unit change in hp (column index 1)
X_hi = X.copy()
X_lo = X.copy()
X_hi[:, 1] = X[:, 1] + 1
X_lo[:, 1] = X[:, 1]

result_comp = comparisons(beta, X_hi, X_lo, vcov, ComparisonType.DIFFERENCE)

print(f"Estimate shape: {result_comp['estimate'].shape}")
print(f"Jacobian shape: {result_comp['jacobian'].shape}")
print(f"Std error shape: {result_comp['std_error'].shape}")
print()

results_comp_df = pl.DataFrame({
    "estimate": result_comp["estimate"][:5],
    "std_error": result_comp["std_error"][:5]
})
print(results_comp_df)

print("\n" + "=" * 60)
print("Linear: Mean comparison")
print("=" * 60)

result_comp_byT = comparisons_byT(beta, X_hi, X_lo, vcov, ComparisonType.DIFFERENCE)

print(f"Mean estimate: {result_comp_byT['estimate']}")
print(f"Standard error: {result_comp_byT['std_error']}")
print(f"Jacobian shape: {result_comp_byT['jacobian'].shape}")

print("\n" + "=" * 60)
print("Linear: Group-level comparisons")
print("=" * 60)

result_comp_byG = comparisons_byG(beta, X_hi, X_lo, vcov, groups, num_groups, ComparisonType.DIFFERENCE)

results_comp_byG_df = pl.DataFrame({
    "cylinder": unique_cyls,
    "estimate": result_comp_byG["estimate"],
    "std_error": result_comp_byG["std_error"]
})
print(results_comp_byG_df)

print("\n" + "=" * 60)
print("GLM: Unit-level comparisons")
print("=" * 60)

X_hi_logit = X_logit.copy()
X_lo_logit = X_logit.copy()
X_hi_logit[:, 1] = X_logit[:, 1] + 1
X_lo_logit[:, 1] = X_logit[:, 1]

result_glm_comp = glm_comparisons.comparisons(
    beta_logit, X_hi_logit, X_lo_logit, vcov_logit,
    ComparisonType.DIFFERENCE, Family.BINOMIAL, Link.LOGIT
)

print(f"Estimate shape: {result_glm_comp['estimate'].shape}")
print(f"Jacobian shape: {result_glm_comp['jacobian'].shape}")
print(f"Std error shape: {result_glm_comp['std_error'].shape}")
print()

results_glm_comp_df = pl.DataFrame({
    "estimate": result_glm_comp["estimate"][:5],
    "std_error": result_glm_comp["std_error"][:5]
})
print(results_glm_comp_df)

print("\n" + "=" * 60)
print("GLM: Mean comparison")
print("=" * 60)

result_glm_comp_byT = glm_comparisons.comparisons_byT(
    beta_logit, X_hi_logit, X_lo_logit, vcov_logit,
    ComparisonType.DIFFERENCE, Family.BINOMIAL, Link.LOGIT
)

print(f"Mean estimate: {result_glm_comp_byT['estimate']}")
print(f"Standard error: {result_glm_comp_byT['std_error']}")
print(f"Jacobian shape: {result_glm_comp_byT['jacobian'].shape}")

print("\n" + "=" * 60)
print("GLM: Group-level comparisons")
print("=" * 60)

result_glm_comp_byG = glm_comparisons.comparisons_byG(
    beta_logit, X_hi_logit, X_lo_logit, vcov_logit,
    groups, num_groups,
    ComparisonType.DIFFERENCE, Family.BINOMIAL, Link.LOGIT
)

results_glm_comp_byG_df = pl.DataFrame({
    "cylinder": unique_cyls,
    "estimate": result_glm_comp_byG["estimate"],
    "std_error": result_glm_comp_byG["std_error"]
})
print(results_glm_comp_byG_df)
