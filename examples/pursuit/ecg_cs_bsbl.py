"""
.. _gallery:cs:ecg:bsbl:1:

ECG Data Compressive Sensing
=====================================

.. contents::
    :depth: 2
    :local:

In this example, we demonstrate the compressive sensing of ECG data
and reconstruction using Block Sparse Bayesian Learning (BSBL).
"""

# Configure JAX to work with 64-bit floating point precision. 
from jax.config import config
config.update("jax_enable_x64", True)


# %% 
# Let's import necessary libraries
import timeit
import jax 
import numpy as np
import jax.numpy as jnp
# CR-Suite libraries
import cr.nimble as crn
import cr.nimble.dsp as crdsp
import cr.sparse.dict as crdict
import cr.sparse.block.bsbl as bsbl

# Sample data
from scipy.misc import electrocardiogram
# Plotting
import matplotlib.pyplot as plt
# Miscellaneous
from scipy.signal import detrend, butter, filtfilt


# %% 
# Test signal
# ------------------------------
# SciPy includes a test electrocardiogram signal
# which is a 5 minute long electrocardiogram (ECG), 
# a medical recording of the electrical activity of the heart, 
# sampled at 360 Hz.
ecg = electrocardiogram()
# Sampling frequency in Hz
fs = 360
# We shall only process a part of the signal in this demo
N = 400
x = ecg[:N]
t = np.arange(N) * (1/fs)
fig, ax = plt.subplots(figsize=(16,4))
ax.plot(t, x);

# %% 
# Preprocessing

# Remove the linear trend from the signal
x = detrend(x)
## bandpass filter
# lower cutoff frequency
f1 = 5
# upper cutoff frequency
f2 = 40
# passband in normalized frequency
Wn = np.array([f1, f2]) * 2 / fs
# butterworth filter
fn = 3
fb, fa = butter(fn, Wn, 'bandpass')
x = filtfilt(fb,fa,x)
fig, ax = plt.subplots(figsize=(16,4))
ax.plot(t, x);



# %% 
# Compressive Sensing at 70%
# ------------------------------
# We choose the compression ratio (M/N) to be 0.7
CR = 0.70
M = int(N * CR)
print(f'M={M}, N={N}, CR={CR}')

# %%
# Sensing matrix
Phi = crdict.gaussian_mtx(crn.KEY0, M, N)

# %%
# Measurements
y = Phi @ x
fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(y);


# %% 
# Sparse Recovery with BSBL
# ------------------------------
options = bsbl.bsbl_bo_options(y, max_iters=20)
start = timeit.default_timer()
sol = bsbl.bsbl_bo_np_jit(Phi, y, 25, options=options)
stop = timeit.default_timer()
print(f'Reconstruction time: {stop - start:.2f} sec', )
print(sol)

# %%
# Recovered signal
x_hat = sol.x
print(f'SNR: {crn.signal_noise_ratio(x, x_hat):.2f} dB, PRD: {crn.percent_rms_diff(x, x_hat):.1f}%')

# %%
# Plot the original and recovered signals
fig, ax = plt.subplots(2, 1, figsize=(16, 4))
ax[0].plot(x)
ax[1].plot(x_hat)


# %% 
# Compressive Sensing at 50%
# ------------------------------
# Let us now increase the compression
CR = 0.50
M = int(N * CR)
print(f'M={M}, N={N}, CR={CR}')

# %%
# Sensing matrix
Phi = crdict.gaussian_mtx(crn.KEY0, M, N)

# %%
# Measurements
y = Phi @ x
fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(y);


# %% 
# Sparse Recovery with BSBL
# ------------------------------
options = bsbl.bsbl_bo_options(y, max_iters=20)
start = timeit.default_timer()
sol = bsbl.bsbl_bo_np_jit(Phi, y, 25, options=options)
stop = timeit.default_timer()
print(f'Reconstruction time: {stop - start:.2f} sec', )
print(sol)

# %%
# Recovered signal
x_hat = sol.x
print(f'SNR: {crn.signal_noise_ratio(x, x_hat):.2f} dB, PRD: {crn.percent_rms_diff(x, x_hat):.1f}%')

# %%
# Plot the original and recovered signals
fig, ax = plt.subplots(2, 1, figsize=(16, 4))
ax[0].plot(x)
ax[1].plot(x_hat)
