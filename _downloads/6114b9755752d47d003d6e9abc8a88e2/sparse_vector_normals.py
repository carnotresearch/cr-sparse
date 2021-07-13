"""
A sparse model vector generator
=================================


Demonstrates how to create sparse model vectors with small number of non-zero entries sampled from Gaussian distribution
"""

# %%
# Let's import necessary libraries 
import matplotlib as mpl
import matplotlib.pyplot as plt
from jax import random
import jax.numpy as jnp
import cr.sparse as crs
import cr.sparse.data as crdata

# %%
# Let's define the size of model and number of sparse entries

# Model size
N = 1000
# Number of non-zero entries in the sparse model
K = 30

# %% 
# Let's generate a random sparse model
key = random.PRNGKey(1)
x, omega = crdata.sparse_normal_representations(key, N, K, 1)
x = jnp.squeeze(x)

# %% 
# We can easily find the locations of non-zero entries
print(crs.nonzero_indices(x))

# %%
# We can  extract corresponding non-zero values in a compact vector
print(crs.nonzero_values(x))

# %% 
# Let's plot the vector to see where the non-zero entries are
plt.figure(figsize=(8,6), dpi= 100, facecolor='w', edgecolor='k')
plt.stem(x, markerfmt='.');
