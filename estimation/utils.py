import jax.numpy as jnp
from jax import jit
import numpy as np
from typing import Union

def standard_errors(J: Union[np.ndarray, jnp.ndarray], V: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
    J = jnp.asarray(J)
    V = jnp.asarray(V)
    return jnp.sqrt(jnp.sum((J @ V) * J, axis=1))

# JIT compiled version
standard_errors_jit = jit(standard_errors)