"""
.. _gallery:cluster:ssc:omp:

Sparse Subspace Clustering - OMP
=============================================

This example demonstrates the sparse subspace clustering algorithm via orthogonal matching pursuit.
"""

# %% 
# Configure JAX to work with 64-bit floating point precision. 
from jax.config import config
config.update("jax_enable_x64", True)

# %% 
# Let's import necessary libraries 
from jax import random
import jax.numpy as jnp
import cr.sparse as crs
import cr.sparse.data as crdata
import cr.sparse.la as crla
import cr.sparse.la.subspaces
# clustering related
import cr.sparse.cluster.spectral as spectral
import cr.sparse.cluster.ssc as ssc
# Plotting
import matplotlib.pyplot as plt
# evaluation
import sklearn.metrics
# Some PRNGKeys for later use
key = random.PRNGKey(0)
keys = random.split(key, 10)

# %% 
# Problem configuration

# ambient space dimension
N = 40
# Subspace dimension
D = 5
# Number of subspaces
K = 5
# Number of points per subspace
S = 50

# %% 
# Test data preparation
# ----------------------------

# %%
# Construct orthonormal bases for K subspaces
bases = crdata.random_subspaces_jit(keys[0], N, D, K)

# %%
# Measure angles between subspaces in degrees
angles = crla.subspaces.smallest_principal_angles_deg(bases)

# %%
# Print the minimum angle between any pair of subspaces
print(crs.off_diagonal_min(angles))

# %%
# Generate uniformly distributed points on each subspace
X = crdata.uniform_points_on_subspaces(keys[1], bases, S)

# %%
# Assign true labels to each point to corresponding 
# subspace index
true_labels = jnp.repeat(jnp.arange(K), S)
print(true_labels)

# %%
# Total number of data points
total = len(true_labels)
print(total)

# %%
# Sparse Subspace Clustering Algorithm
# ------------------------------------------

# %%
# Build representation of each point in terms of other points
# by using Orthogonal Matching Pursuit algorithm
Z, I, R = ssc.build_representation_omp_jit(X, D)

# %%
# Combine values and indices to form full representation
Z_full = ssc.sparse_to_full_rep(Z, I)

# %%
# Build the affinity matrix
affinity = abs(Z_full) + abs(Z_full).T
plt.imshow(affinity, cmap='gray')

# %%
# Perform the spectral clustering on the affinity matrix
res = spectral.unnormalized_k_jit(keys[2], affinity, K)

# %%
# Predicted cluster labels
pred_labels = res.assignment
print(pred_labels)

# %%
# Evaluate the clustering performance
print(sklearn.metrics.rand_score(true_labels, pred_labels))



# %%
# SSC-OMP with shuffled data
# ------------------------------------------

# %%
# Choose a random permutation
perm = random.permutation(keys[3], total)

# %%
# Randomly permute the data points
X = X[:, perm]
# Permute the true labels accordingly
true_labels = true_labels[perm]
print(true_labels)

# %%
# Build representation of each point in terms of other points
# by using Orthogonal Matching Pursuit algorithm
Z, I, R = ssc.build_representation_omp_jit(X, D)

# %%
# Combine values and indices to form full representation
Z_full = ssc.sparse_to_full_rep(Z, I)

# %%
# Build the affinity matrix
affinity = abs(Z_full) + abs(Z_full).T
plt.imshow(affinity, cmap='gray')

# %%
# Perform the spectral clustering on the affinity matrix
res = spectral.unnormalized_k_jit(keys[4], affinity, K)

# %%
# Predicted cluster labels
pred_labels = res.assignment
print(pred_labels)

# %%
# Evaluate the clustering performance
print(sklearn.metrics.rand_score(true_labels, pred_labels))

