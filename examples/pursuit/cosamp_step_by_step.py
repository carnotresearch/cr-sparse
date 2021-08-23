"""
CoSaMP step by step
==========================

This example explains the step by step development of 
CoSaMP (Compressive Sensing Matching Pursuit) algorithm
for sparse recovery. It then shows how to use the 
official implementation of CoSaMP in ``CR-Sparse``.


The CoSaMP algorithm has following inputs:

* A sensing matrix or dictionary ``Phi`` which has been used for data measurements.
* A measurement vector ``y``.
* The sparsity level ``K``.

The objective of the algorithm is to estimate a K-sparse solution ``x``
such that ``y`` is approximately equal to ``Phi x``.

A key quantity in the algorithm is the residual ``r = y - Phi x``. Each 
iteration of the algorithm successively improves the estimate ``x`` so 
that the energy of the residual ``r`` reduces.

The algorithm proceeds as follows:

* Initialize the solution ``x`` with zero.
* Maintain an index set ``I`` (initially empty) of atoms selected as part of the solution.
* While the residual energy is above a threshold:

 * **Match**: Compute the inner product of each atom in ``Phi`` with the current residual ``r``.
 * **Identify**: Select the indices of 2K atoms from ``Phi`` with the largest correlation with the residual.
 * **Merge**: merge these 2K indices with currently selected indices in ``I`` to form ``I_sub``. 
 * **LS**: Compute the least squares solution of ``Phi[:, I_sub] z = y``
 * **Prune**: Pick the largest K entries from this least square solution and keep them in ``I``. 
 * **Update residual**: Compute ``r = y - Phi_I x_I``.

It is time to see the algorithm in action.
"""

# %% 
# Let's import necessary libraries 
import jax
from jax import random
import jax.numpy as jnp
# Some keys for generating random numbers
key = random.PRNGKey(0)
keys = random.split(key, 4)

# For plotting diagrams
import matplotlib.pyplot as plt
# CR-Sparse modules
import cr.sparse as crs
import cr.sparse.dict as crdict
import cr.sparse.data as crdata

# %%
# Problem Setup
# ------------------

# Number of measurements
M = 128
# Ambient dimension
N = 256
# Sparsity level
K = 8

# %%
# The Sparsifying Basis
# ''''''''''''''''''''''''''
Phi = crdict.gaussian_mtx(key, M,N)
print(Phi.shape)

# %%
# Coherence of atoms in the sensing matrix
print(crdict.coherence(Phi))

# %%
# A sparse model vector
# ''''''''''''''''''''''''''
x0, omega = crdata.sparse_normal_representations(key, N, K)
plt.figure(figsize=(8,6), dpi= 100, facecolor='w', edgecolor='k')
plt.plot(x0)

# %%
# ``omega`` contains the set of indices at which x is nonzero (support of ``x``)
print(omega)

# %%
# Compressive measurements
# ''''''''''''''''''''''''''
y = Phi @ x0
plt.figure(figsize=(8,6), dpi= 100, facecolor='w', edgecolor='k')
plt.plot(y)

# %%
# Development of CoSaMP algorithm
# ---------------------------------

# In the following, we walk through the steps of CoSaMP algorithm.
# Since we have access to ``x0`` and ``omega``, we can measure the
# progress made by the algorithm steps by comparing the estimates
# with actual ``x0`` and ``omega``. However, note that in the 
# real implementation of the algorithm, no access to original model
# vector is there.
#
# Initialization
# ''''''''''''''''''''''''''''''''''''''''''''

# %%
# We assume the initial solution to be zero and 
# the residual ``r = y - Phi x`` to equal the measurements ``y``
r = y
# %%
# Squared norm/energy of the residual
y_norm_sqr = float(y.T @ y)
r_norm_sqr = y_norm_sqr
print(f"{r_norm_sqr=}")

# %%
# A boolean array to track the indices selected for least squares steps
flags = jnp.zeros(N, dtype=bool)
# %%
# During the matching steps, 2K atoms will be picked.
K2 = 2*K
# %%
# At any time, up to 3K atoms may be selected (after the merge step).
K3 = K + K2

# %%
# Number of iterations completed so far
iterations = 0


# %%
# A limit on the maximum tolerance for residual norm
res_norm_rtol = 1e-3
max_r_norm_sqr = y_norm_sqr * (res_norm_rtol ** 2)
print(f"{max_r_norm_sqr=:.2e}")

# %%
# First iteration
# ''''''''''''''''''''''''''''''''''''''''''''
print("First iteration:")
# %%
# Match the current residual with the atoms in ``Phi``
h = Phi.T @ r

# %%
# Pick the indices of 3K atoms with largest matches with the residual
I_sub =  crs.largest_indices(h, K3)
# Update the flags array
flags = flags.at[I_sub].set(True)
# Sort the ``I_sub`` array with the help of flags array
I_sub, = jnp.where(flags)
# Since no atoms have been selected so far, we can be more aggressive
# and pick 3K atoms in first iteration. 
print(f"{I_sub=}")
# %%
# Check which indices from ``omega`` are there in ``I_sub``.
print(jnp.intersect1d(omega, I_sub))
# %%
# Select the subdictionary of ``Phi`` consisting of atoms indexed by I_sub
Phi_sub = Phi[:, flags]
# %%
# Compute the least squares solution of ``y`` over this subdictionary
x_sub, r_sub_norms, rank_sub, s_sub = jnp.linalg.lstsq(Phi_sub, y)
# Pick the indices of K largest entries in in ``x_sub`` 
Ia = crs.largest_indices(x_sub, K)
print(f"{Ia=}")
# %%
# We need to map the indices in ``Ia`` to the actual indices of atoms in ``Phi``
I = I_sub[Ia]
print(f"{I=}")
# %%
# Select the corresponding values from the LS solution
x_I = x_sub[Ia]
# %%
# We now have our first estimate of the solution
x = jnp.zeros(N).at[I].set(x_I)
plt.figure(figsize=(8,6), dpi= 100, facecolor='w', edgecolor='k')
plt.plot(x0, label="Original vector")
plt.plot(x, '--', label="Estimated solution")
plt.legend()
# %%
# We can check how good we were in picking the correct indices from the actual support of the signal
found = jnp.intersect1d(omega, I)
print("Found indices: ", found)
# %%
# We found 6 out of 8 indices in the support. Here are the remaining.
missing = jnp.setdiff1d(omega, I)
print("Missing indices: ", missing)
# %%
# It is time to compute the residual after the first iteration
Phi_I = Phi[:, I]
r = y - Phi_I @ x_I
# %%
# Compute the residual and verify that it is still larger than the allowed tolerance
r_norm_sqr = float(r.T @ r)
print(f"{r_norm_sqr=:.2e} > {max_r_norm_sqr=:.2e}")
# %% 
# Store the selected K indices in the flags array
flags = flags.at[:].set(False)
flags = flags.at[I].set(True)
print(jnp.where(flags))
# %%
# Mark the completion of the iteration
iterations += 1

# %%
# Second iteration
# ''''''''''''''''''''''''''''''''''''''''''''
print("Second iteration:")
# %%
# Match the current residual with the atoms in ``Phi``
h = Phi.T @ r
# %%
# Pick the indices of 2K atoms with largest matches with the residual
I_2k =  crs.largest_indices(h, K2 if iterations else K3)
# We can check if these include the atoms missed out in first iteration.
print(jnp.intersect1d(omega, I_2k))
# %%
# Merge (union) the set of previous K indices with the new 2K indices
flags = flags.at[I_2k].set(True)
I_sub, = jnp.where(flags)
print(f"{I_sub=}")
# %%
# We can check if we found all the actual atoms
print("Found in I_sub: ", jnp.intersect1d(omega, I_sub))
# %%
# Indeed we did. The set difference is empty.
print("Missing in I_sub: ", jnp.setdiff1d(omega, I_sub))

# %% 
# Select the subdictionary of ``Phi`` consisting of atoms indexed by ``I_sub``
Phi_sub = Phi[:, flags]
# %%
# Compute the least squares solution of ``y`` over this subdictionary
x_sub, r_sub_norms, rank_sub, s_sub = jnp.linalg.lstsq(Phi_sub, y)
# Pick the indices of K largest entries in in ``x_sub`` 
Ia = crs.largest_indices(x_sub, K)
print(Ia)
# %%
# We need to map the indices in ``Ia`` to the actual indices of atoms in ``Phi``
I = I_sub[Ia]
print(I)
# %%
# Check if the final K indices in ``I`` include all the indices in ``omega``
jnp.setdiff1d(omega, I)
# %%
# Select the corresponding values from the LS solution
x_I = x_sub[Ia]
# %%
# Here is our updated estimate of the solution
x = jnp.zeros(N).at[I].set(x_I)
plt.figure(figsize=(8,6), dpi= 100, facecolor='w', edgecolor='k')
plt.plot(x0, label="Original vector")
plt.plot(x, '--', label="Estimated solution")
plt.legend()
# %%
# The algorithm has no direct way of knowing that it indeed found the solution
# It is time to compute the residual after the second iteration
Phi_I = Phi[:, I]
r = y - Phi_I @ x_I
# %%
# Compute the residual and verify that it is now below the allowed tolerance
r_norm_sqr = float(r.T @ r)
# It turns out that it is now below the tolerance threshold
print(f"{r_norm_sqr=:.2e} < {max_r_norm_sqr=:.2e}")
# %%
# We have completed the signal recovery. We can stop iterating now.
iterations += 1

# %%
# CR-Sparse official implementation
# ----------------------------------------
# The JIT compiled version of this algorithm is available in 
# ``cr.sparse.pursuit.cosamp`` module.

# %%
# Import the module
from cr.sparse.pursuit import cosamp
# %%
# Run the solver
solution =  cosamp.matrix_solve_jit(Phi, y, K)
# The support for the sparse solution
I = solution.I
print(I)
# %%
# The non-zero values on the support
x_I = solution.x_I
print(x_I)
# %%
# Verify that we successfully recovered the support
print(jnp.setdiff1d(omega, I))
# %% 
# Print the residual energy and the number of iterations when the algorithm converged.
print(solution.r_norm_sqr, solution.iterations)
# %%
# Let's plot the solution
x = jnp.zeros(N).at[I].set(x_I)
plt.figure(figsize=(8,6), dpi= 100, facecolor='w', edgecolor='k')
plt.plot(x0, label="Original vector")
plt.plot(x, '--', label="Estimated solution")
plt.legend()
