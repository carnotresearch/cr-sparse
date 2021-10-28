"""
Frequency Change Detection
===============================

This example considers a signal which is a sinusoid whose
frequency changes abruptly at a specific point in time.
Looking at the waveform of the signal, it is very difficult
to identify the time where the frequency change occurs.
However, when we perform the wavelet decomposition of the
signal, we can clearly see the location at which the
discontinuity in frequency occurs.

We will do the following:

- Synthetically construct a sinusoid with two different frequencies
  at different times.
- Examine its Fourier spectrum which cannot reveal this discontinuity.
- Perform multilevel wavelet decomposition.
- Visually inspect the detail coefficients at multiple levels to 
  locate the discontinuity
- Algorithmically locate the discontinuity by scanning the detail 
  coefficients.
"""

# Configure JAX to work with 64-bit floating point precision. 
from jax.config import config
config.update("jax_enable_x64", True)
# %% 
# Let's import necessary libraries 
import jax.numpy as jnp
# CR-Sparse libraries
import cr.sparse as crs
import cr.sparse.wt as wt
# Utilty functions to construct sinusoids
import cr.sparse.dsp.signals as signals
# Plotting
import matplotlib.pyplot as plt

# %% 
# Test signal generation
# ------------------------------

# Sampling frequency in Hz
fs = 100
# Signal duration in seconds
T = 20
# Frequency of first part of signal in Hz
f = 1/8
# Start and end times of first part of signal
start_time = 0
end_time = 12
# Construct the first part of signal
t, x1 = signals.transient_sine_wave(fs, T, f, start_time, end_time)
# Frequency of second part of signal in Hz
f = 1/6
# Start and end times of first part of signal
start_time = end_time
end_time = T
# Adjust the initial phase of second part of signal to be in continuity with the first part
initial_phase=jnp.pi
# Construct the second part of signal
t, x2 = signals.transient_sine_wave(fs, T, f, start_time, end_time, initial_phase=initial_phase)
# Combine the first and second parts of signal
x = x1 + x2
# Overall signal length
n = len(x)
# Plot the parts and combined signal
fig, axs = plt.subplots(3, figsize=(12,12))
axs[0].plot(t,x1)
axs[1].plot(t,x2)
axs[2].plot(t,x)
# %%
# The last plot shows the combined signal with discontinuity at t=12 sec when
# the frequency changes from 1/8 Hz to 1/6 Hz.
# The change is very difficult to notice as well as locate in the time domain plot of the signal.

# %% 
# Frequency spectrum
# ------------------------------

# Compute the FFT
f = jnp.fft.fftshift(jnp.fft.fft(x))
# Plot the magnitude
fig, axs = plt.subplots(1, figsize=(12,4))
plt.plot(jnp.abs(f[750:1250]))
# %% 
# The frequencies 1/6 and 1/8 Hz are so close that it is difficult to distinguish them 
# in the frequency spectrum. Of course, location of the discontinuity cannot be 
# identified in the frequency spectrum.


# %% 
# Multilevel wavelet decomposition
# ------------------------------

# Compute the multilevel wavelet decomposition
coeffs = wt.wavedec(x, 'db4')
# Total number of decomposition levels
levels = len(coeffs) -1
print(levels)

# %% 
# First level detail coefficients
cd1 = coeffs[-1]
# Dyadic upsample them to align with the time values
cd1 = wt.dyadup_in(cd1)
# %% 
# Second level detail coefficients
cd2 = coeffs[-2]
# Dyadic upsample them to align with the time values
cd2 = wt.dyadup_in(wt.dyadup_in(cd2))
# Plot the first and second level detail coefficients
fig, axs = plt.subplots(2, figsize=(12,12))
axs[0].plot(t, cd1[:n])
axs[1].plot(t, cd2[:n])
# %%
# The discontinuities are clearly visible in the plots of detail coefficients 
# both at first level and second level.
# The detail coefficients have been aligned with the time values.
# Both plots should large coefficients around t=12 sec.
# Large coefficients at the boundary are due to boundary effects in DWT and can be safely ignored.

# %% 
# Locate the indices of largest entries (by magnitude) in the detail 
# coefficients at first level
idx, values = crs.hard_threshold_sorted(cd1, 8)
print(idx)
# %%
# After ignoring the first couple of entries for the boundary effects
# at the beginning of the data, we can see that the discontinuity
# has been correctly identified at sample 1200 which happens to 
# correspond to t=12 sec.