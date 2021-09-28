"""
.. _gallery:lop:cs_operators:

Compressive Sensing Operators
===============================================

.. contents::
    :depth: 2
    :local:

"""

# %% 
# Let's import necessary libraries 
import jax.numpy as jnp
# Random numbers
from jax import random
# For plotting diagrams
import matplotlib.pyplot as plt
## CR-Sparse modules
import cr.sparse as crs
# Linear operators
from cr.sparse import lop
# Some random number seeds
key = random.PRNGKey(0)
keys = random.split(key, 10)


# %%
# Rademacher sensing operators
# -----------------------------------------------------------------
# Size of the sensing matrix
m, n = 4, 8
# %%
# Unnormalized atoms / columns
# '''''''''''''''''''''''''''''''''''
# Construct the operator wrapping the sensing matrix
Phi = lop.rademacher_dict(keys[0], m, n, normalize_atoms=False)
# %%
# Let's print the contents of the matrix
print(lop.to_matrix(Phi))
# %%
# Let's print the contents of the adjoint matrix
print(lop.to_adjoint_matrix(Phi))
# %%
# Let's apply the forward operator
x = jnp.array([3, -2, 4, 8, 2, 2, 6, -1])
y = Phi.times(x)
print(y)
# %%
# Let's apply the adjoint operator
z = Phi.trans(y)
print(z)
# %%
# Let's JIT compile the trans and times functions
Phi = lop.jit(Phi)
# %%
# Let's verify the forward operator after JIT compile
x = jnp.array([3, -2, 4, 8, 2, 2, 6, -1])
y = Phi.times(x)
print(y)
# %%
# Let's verify the adjoint operator after JIT compile
z = Phi.trans(y)
print(z)

# %%
# Normalized atoms / columns
# '''''''''''''''''''''''''''''''''''
# Construct the operator wrapping the sensing matrix
Phi = lop.rademacher_dict(keys[0], m, n, normalize_atoms=True)
# %%
# Let's print the contents of the matrix
print(lop.to_matrix(Phi))
# %%
# Check the column wise norms
print(crs.norms_l2_cw(lop.to_matrix(Phi)))
# %%
# Let's print the contents of the adjoint matrix
print(lop.to_adjoint_matrix(Phi))
# %%
# Let's apply the forward operator
x = jnp.array([3, -2, 4, 8, 2, 2, 6, -1])
y = Phi.times(x)
print(y)
# %%
# Let's apply the adjoint operator
z = Phi.trans(y)
print(z)
# %%
# Let's JIT compile the trans and times functions
Phi = lop.jit(Phi)
# %%
# Let's verify the forward operator after JIT compile
y = Phi.times(x)
print(y)
# %%
# Let's verify the adjoint operator after JIT compile
z = Phi.trans(y)
print(z)
# %%
# Column wise application on input
# '''''''''''''''''''''''''''''''''''''''
# Number of signals
k = 5
# %%
# Prepare a signal matrix
X = random.randint(keys[1], (n, k), -5, 5)
print(X)
# %%
# Let's apply the forward operator column wise
Y = Phi.times(X)
print(Y)
# %%
# Let's apply the adjoint operator column wise
Z = Phi.trans(Y)
print(Z)

# %%
# Row wise application on input
# '''''''''''''''''''''''''''''''''''''''
# Construct the operator wrapping the sensing matrix
Phi = lop.rademacher_dict(keys[0], m, n, normalize_atoms=True, axis=1)
# %%
# Prepare a signal matrix with each signal along a row
XT = X.T
# %%
# Let's apply the forward operator column wise
Y = Phi.times(XT)
print(Y)
# %%
# Let's apply the adjoint operator column wise
Z = Phi.trans(Y)
print(Z)

# %%
# Square sensing matrix
# '''''''''''''''''''''''''''''''''''
# Construct the operator wrapping the sensing matrix
Phi = lop.rademacher_dict(keys[0], n, normalize_atoms=False)
# %%
# Let's print the contents of the matrix
print(lop.to_matrix(Phi))


# %%
# Gaussian sensing operators
# -----------------------------------------------------------------

# %%
# Unnormalized atoms / columns
# '''''''''''''''''''''''''''''''''''
# Construct the operator wrapping the sensing matrix
Phi = lop.gaussian_dict(keys[0], m, n, normalize_atoms=False)
# %%
# Let's print the contents of the matrix
print(lop.to_matrix(Phi))
# %%
# Let's print the contents of the adjoint matrix
print(lop.to_adjoint_matrix(Phi))
# %%
# Let's apply the forward operator
x = jnp.array([3, -2, 4, 8, 2, 2, 6, -1])
y = Phi.times(x)
print(y)
# %%
# Let's apply the adjoint operator
z = Phi.trans(y)
print(z)
# %%
# Let's JIT compile the trans and times functions
Phi = lop.jit(Phi)
# %%
# Let's verify the forward operator after JIT compile
x = jnp.array([3, -2, 4, 8, 2, 2, 6, -1])
y = Phi.times(x)
print(y)
# %%
# Let's verify the adjoint operator after JIT compile
z = Phi.trans(y)
print(z)

# %%
# Normalized atoms / columns
# '''''''''''''''''''''''''''''''''''
# Construct the operator wrapping the sensing matrix
Phi = lop.gaussian_dict(keys[0], m, n, normalize_atoms=True)
# %%
# Let's print the contents of the matrix
print(lop.to_matrix(Phi))
# %%
# Check the column wise norms
print(crs.norms_l2_cw(lop.to_matrix(Phi)))
# %%
# Let's print the contents of the adjoint matrix
print(lop.to_adjoint_matrix(Phi))
# %%
# Let's apply the forward operator
x = jnp.array([3, -2, 4, 8, 2, 2, 6, -1])
y = Phi.times(x)
print(y)
# %%
# Let's apply the adjoint operator
z = Phi.trans(y)
print(z)
# %%
# Let's JIT compile the trans and times functions
Phi = lop.jit(Phi)
# %%
# Let's verify the forward operator after JIT compile
y = Phi.times(x)
print(y)
# %%
# Let's verify the adjoint operator after JIT compile
z = Phi.trans(y)
print(z)
# %%
# Column wise application on input
# '''''''''''''''''''''''''''''''''''''''
# Number of signals
k = 5
# %%
# Prepare a signal matrix
X = random.randint(keys[1], (n, k), -5, 5)
print(X)
# %%
# Let's apply the forward operator column wise
Y = Phi.times(X)
print(Y)
# %%
# Let's apply the adjoint operator column wise
Z = Phi.trans(Y)
print(Z)

# %%
# Row wise application on input
# '''''''''''''''''''''''''''''''''''''''
# Construct the operator wrapping the sensing matrix
Phi = lop.gaussian_dict(keys[0], m, n, normalize_atoms=True, axis=1)
# %%
# Prepare a signal matrix with each signal along a row
XT = X.T
# %%
# Let's apply the forward operator column wise
Y = Phi.times(XT)
print(Y)
# %%
# Let's apply the adjoint operator column wise
Z = Phi.trans(Y)
print(Z)

# %%
# Square sensing matrix
# '''''''''''''''''''''''''''''''''''
# Construct the operator wrapping the sensing matrix
Phi = lop.gaussian_dict(keys[0], n, normalize_atoms=False)
# %%
# Let's print the contents of the matrix
print(lop.to_matrix(Phi))


# %%
# Random matrix with orthonormal rows
# -----------------------------------------------------------------
# Construct the operator wrapping the sensing matrix
Phi = lop.random_orthonormal_rows_dict(keys[0], m, n)
# %%
# Let's print the contents of the matrix
print(lop.to_matrix(Phi))
# %%
# Check the row wise norms
print(crs.norms_l2_rw(lop.to_matrix(Phi)))
# %%
# Let's apply the forward operator column wise
Y = Phi.times(X)
print(Y)
# %%
# Let's apply the adjoint operator column wise
Z = Phi.trans(Y)
print(Z)
# %%
# The frame operator
F = lop.frame(Phi)
# %%
# Check that it's close to identity matrix
print(jnp.round(lop.to_matrix(F),2))

# %%
# Random orthonormal basis operator
# -----------------------------------------------------------------
# Construct the operator wrapping the sensing matrix
Phi = lop.random_onb_dict(keys[0], n)
# %%
# Let's print the contents of the matrix
print(lop.to_matrix(Phi))
# %%
# Check the row wise norms
print(crs.norms_l2_rw(lop.to_matrix(Phi)))
# %%
# Check the column wise norms
print(crs.norms_l2_cw(lop.to_matrix(Phi)))
# %%
# Let's apply the forward operator column wise
Y = Phi.times(X)
print(Y)
# %%
# Let's apply the adjoint operator column wise
Z = Phi.trans(Y)
print(Z)
# %%
# The gram operator
G = lop.gram(Phi)
# %%
# Check that it's close to identity matrix
print(jnp.round(lop.to_matrix(G),2))
# %%
# The frame operator
F = lop.frame(Phi)
# %%
# Check that it's close to identity matrix
print(jnp.round(lop.to_matrix(F),2))

