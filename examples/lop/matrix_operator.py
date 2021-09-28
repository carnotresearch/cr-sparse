"""
.. _gallery:lop:matrix_op:

A matrix backed linear operator
==================================

This example demonstrates how to construct a linear operator from an ordinary matrix.
"""

# %% 
# Let's import necessary libraries 
import jax.numpy as jnp
from cr.sparse import lop

# %%
# Setup
# ------
# Create a small matrix
n = 4
A = jnp.ones((n, n))
print(A)

# %%
# Create a vector
x = jnp.ones(n)
print(x)

# %%
# Operator construction
# ----------------------
# Convert A into an operator
T = lop.matrix(A)

# %%
# Operator usage
# ----------------------

# %%
# Compute A x
print(T.times(x))

# %%
# Compute A^H x
print(T.trans(x))


# %%
# JIT Compilation
# -----------------
# Wrap the ``times`` and ``trans`` functions with jit
T = lop.jit(T)

# %%
# Compute A x
print(T.times(x))

# %%
# Compute A^H x
print(T.trans(x))


