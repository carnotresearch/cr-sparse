"""
This example demonstrates how to construct a linear operator from an ordinary matrix.
"""

import jax.numpy as jnp
from cr.sparse import lop

# Create a small matrix
n = 4
A = jnp.ones((n, n))
A
# Convert A into an operator
T = lop.matrix(A)
# Create a vector
x = jnp.ones(n)

# Compute A x
T.times(x)

# Compute A^H x
T.trans(x)
