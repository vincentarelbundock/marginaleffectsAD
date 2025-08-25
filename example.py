import numpy as np
import pandas as pd
import polars as pl
from marginaleffects import get_dataset
from formulaic import model_matrix
import statsmodels.api as sm
from estimation import logit
from estimation.utils import standard_errors

# Step 1: Download mtcars dataset
mtcars = get_dataset("mtcars", "datasets").to_pandas()
y, X = model_matrix("am ~ hp * wt + C(cyl)", mtcars)
y = y.values
X = X.values

# Step 3: Fit logit model using statsmodels
logit_result = sm.Logit(y, X).fit()

# Step 4: Extract coefficients
beta = logit_result.params

# Step 5: Make predictions using JAX implementation
jax_predictions = logit.predict(beta, X)

# Step 6: Compute standard errors using JAX
J = logit.jacobian(beta, X)

# Get variance-covariance matrix from statsmodels
V = logit_result.cov_params()

# Compute standard errors using JAX
jax_standard_errors = standard_errors(J, V)

results = pl.DataFrame(
    {"estimate": np.array(jax_predictions), "std_error": np.array(jax_standard_errors)}
)
print(results)
