"""
Wavelet Transform Operators
==============================

.. contents::
    :depth: 2
    :local:

This example demonstrates following features:

- ``cr.sparse.lop.convolve2D`` A 2D convolution linear operator
- ``cr.sparse.sls.lsqr`` LSQR algorithm for solving a least square problem on 2D images


"""

# %% 
# Let's import necessary libraries 
import jax.numpy as jnp
# For plotting diagrams
import matplotlib.pyplot as plt
## CR-Sparse modules
import cr.sparse as crs
# Linear operators
from cr.sparse import lop
# Error measurement
from cr.sparse import metrics
# Sample images
import skimage.data
# Utilities
from cr.sparse.dsp import time_values
# Configure JAX for 64-bit computing
from jax.config import config
config.update("jax_enable_x64", True)

# %%
# 1D Wavelet Transform Operator
# ---------------------------------------------------


# %%
# A signal consisting of multiple sinusoids 
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Individual sinusoids have different frequencies and amplitudes.
# Sampling frequency
fs = 250.
# Time duration
T = 2
# time values
t = time_values(fs, T)
# Number of samples
n = t.size
x = jnp.zeros(n)
freqs = [10, 7, 9]
amps = [1, -2, .5]
for  (f, amp) in zip(freqs, amps):
    sinusoid = amp * jnp.sin(2 * jnp.pi * f * t)
    x = x + sinusoid
# Plot the signal
plt.figure(figsize=(4,2))
plt.plot(t, x, 'k', label='Composite signal')

# %%
# 1D wavelet transform operator
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
DWT_op = lop.dwt(n, wavelet='dmey', level=5)

# %%
# Wavelet coefficients
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
alpha = DWT_op.times(x)
plt.figure(figsize=(4,2))
plt.plot(alpha, 'k', label='Wavelet coefficients')

# %%
# Compression
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Let's keep only 12.5 percent of the coefficients
cutoff = n // 8
alpha2 = alpha.at[:cutoff].set(0)
plt.figure(figsize=(4,2))
plt.plot(alpha2, 'k', label='Wavelet coefficients after compression')

# %%
# Reconstruction
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
x_rec = DWT_op.trans(alpha2)
# RMSE 
rmse = metrics.root_mse(x, x_rec)
print(rmse)
# SNR 
snr = metrics.signal_noise_ratio(x, x_rec)
print(snr)
plt.figure(figsize=(8,2))
plt.plot(x, 'k', label='Original')
plt.plot(x_rec, 'r', label='Reconstructed')
plt.title('Reconstructed signal')
