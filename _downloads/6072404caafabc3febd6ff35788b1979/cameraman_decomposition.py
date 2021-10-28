"""
Wavelet Decomposition of an Image
======================================

This is a simple example which demonstrates how 
``cr.sparse.wt.dwt2`` function can be used to 
perform 2D wavelet decomposition.

Its interface is identical to the corresponding function
in PyWavelets library.

This example is adapted from 
`PyWavelets documentation <https://pywavelets.readthedocs.io/en/latest/>`_.
"""

# %% 
# Configure JAX to work with 64-bit floating point precision. 
from jax.config import config
config.update("jax_enable_x64", True)
# %% 
# Let's import necessary libraries 
import jax.numpy as jnp
# CR-Sparse libraries
import cr.sparse.wt as wt
# We use PyWavelets only for sample data
import pywt.data
# Plotting
import matplotlib.pyplot as plt

# %% 
# Load the Cameraman image
original = pywt.data.camera()
# %% 
# Perform wavelet decomposition 
coeffs2 = wt.dwt2(original, 'bior1.3')
# %% 
# Split the coefficients tuple into individual parts
LL, (LH, HL, HH) = coeffs2
# %% 
# Plot the decomposition
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
