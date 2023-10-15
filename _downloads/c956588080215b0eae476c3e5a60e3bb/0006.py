r"""
.. _gallery:0006:

Piecewise Cubic, Daubechies Basis, Gaussian Measurements
=============================================================

.. contents::
    :depth: 2
    :local:

In this example we have

#. A signal :math:`\by` consisting of piecewise cubic
   polynomials with 5 pieces of total length of 2048 samples.
#. A Daubechies-8 wavelet basis :math:`\Psi` of shape 2048x2048
   with 5 levels of decomposition.
#. The sparse representation :math:`\bx` of the signal :math:`\by`
   in the basis :math:`\Psi` consisting of exactly 63 nonzero
   entries (corresponding to the spikes and the amplitudes of the cosine waves).
#. A Gaussian sensing matrix :math:`\Phi` of shape  600x2048 making
   600 random measurements in a vector :math:`\bb`.
   The columns of the sensing matrix are unit normalized.
#. We are given :math:`\bb` and :math:`\bA = \Phi \Psi` and
   have to reconstruct :math:`\bx` using it.
#. Then we can use :math:`\Psi` to compute :math:`\by = \Psi \bx`.


.. math::

    \bb = \bA \bx = \Phi \Psi \bx = \Phi \by.

See also:

* :ref:`api:problems`
* :ref:`api:lop`
"""

# Configure JAX to work with 64-bit floating point precision. 
from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import cr.nimble as crn
import cr.sparse as crs
import cr.sparse.plots as crplot


# %% 
# Setup
# ------------------------------
# We shall construct our test signal and dictionary
# using our test problems module.

from cr.sparse import problems
prob = problems.generate('piecewise-cubic-poly:daubechies:gaussian')
fig, ax = problems.plot(prob)


# %% 
# Let us access the relevant parts of our test problem

# The combined linear operator (sensing matrix + dictionary)
A = prob.A
# The sparse representation of the signal in the dictionary
x0 = prob.x
# The Cosine+Spikes signal
y0 = prob.y
# The measurements
b0 = prob.b

# %% 
# Check how many coefficients in the sparse representation
# are sufficient to capture 99.9% of the energy of the signal
print(crn.num_largest_coeffs_for_energy_percent(x0, 99.9))

# %% 
# Check how many coefficients in the sparse representation
# are sufficient to capture 100% of the energy of the signal
print(crn.num_largest_coeffs_for_energy_percent(x0, 100))

# %% 
# This number gives us an idea about the required sparsity
# to be configured for greedy pursuit algorithms.
# Although the exact sparsity of this representation is 63
# but several of the spikes are too small and could be ignored
# for a reasonably good approximation.

# %% 
# Sparse Recovery using Subspace Pursuit
# -------------------------------------------
# We shall use subspace pursuit to reconstruct the signal.
import cr.sparse.pursuit.sp as sp

# We will first try to estimate a 100-sparse representation
sol = sp.solve(A, b0, 100)
# %%
# This utility function helps us quickly analyze the quality of reconstruction
problems.analyze_solution(prob, sol, perc=100)

# %%
# We will now try to estimate a 150-sparse representation
sol = sp.solve(A, b0, 150)
# %%
# Let us check if we correctly decoded all the nonzero entries
# in the sparse representation x
problems.analyze_solution(prob, sol, perc=100)


# %%
# We will now try to estimate a 200-sparse representation
sol = sp.solve(A, b0, 200)
# %%
# Let us check if we correctly decoded all the nonzero entries
# in the sparse representation x
problems.analyze_solution(prob, sol, perc=100)

# %%
# We will now try to estimate a 250-sparse representation
tracker = crs.ProgressTracker(x0=x0)
sol = sp.solve(A, b0, 250, tracker=tracker)

# %% 
# Let us plot the progress of subspace pursuit over different iterations
ax = crplot.one_plot(height=6)
tracker.plot_progress(ax)

# %%
# Let us check if we correctly decoded all the nonzero entries
# in the sparse representation x
problems.analyze_solution(prob, sol, perc=100)

# %% 
# The estimated sparse representation
x = sol.x
# %%
# Let us reconstruct the signal from this sparse representation
y = prob.reconstruct(x)
# %%
# The estimated measurements
b = A.times(x)

# %%
# Let us visualize the original and reconstructed representation
def plot_representations(x0, x):
    ax = crplot.h_plots(2, height=2)
    ax[0].stem(x0, markerfmt='.')
    ax[0].set_title('Original representation')
    ax[1].stem(x, markerfmt='.')
    ax[1].set_title('Reconstructed representation')
plot_representations(x0, x)


# %%
# Let us visualize the original and reconstructed signal
def plot_signals(y0, y):
    ax = crplot.h_plots(2, height=2)
    ax[0].plot(y0)
    ax[0].set_title('Original signal')
    ax[1].plot(y)
    ax[1].set_title('Reconstructed signal')
plot_signals(y0, y)

# %%
# Let us visualize the original and reconstructed measurements
def plot_measurments(b0, b):
    ax = crplot.h_plots(2, height=2)
    ax[0].plot(b0)
    ax[0].set_title('Original measurements')
    ax[1].plot(b)
    ax[1].set_title('Reconstructed measurements')
plot_measurments(b0, b)


# %%
# We will now try to estimate a 278-sparse representation
sol = sp.solve(A, b0, 278)
# %%
# Let us check if we correctly decoded all the nonzero entries
# in the sparse representation x
problems.analyze_solution(prob, sol, perc=100)



# %% 
# Sparse Recovery using SPGL1
# ---------------------------------------------------------------
import cr.sparse.cvx.spgl1 as crspgl1
sigma = 0.01 * jnp.linalg.norm(b0)
options = crspgl1.SPGL1Options(max_iters=1000)
tracker = crs.ProgressTracker(x0=x0, every=5)
sol = crspgl1.solve_bpic_jit(A, b0, sigma, 
    options=options, tracker=tracker)
# %%
# Analyze the solution
problems.analyze_solution(prob, sol, perc=100)

# %%
# Try with lower threshold on allowed noise
sigma = 0.001 * jnp.linalg.norm(b0)
options = crspgl1.SPGL1Options(max_iters=1000)
tracker = crs.ProgressTracker(x0=x0, every=20)
sol = crspgl1.solve_bpic_jit(A, b0, sigma, 
    options=options, tracker=tracker)

# %% 
# Let us plot the progress of SPGL1 over different iterations
ax = crplot.one_plot(height=6)
tracker.plot_progress(ax)


# %%
# Analyze the solution
problems.analyze_solution(prob, sol, perc=100)

# %% 
# The estimated sparse representation
x = sol.x
# %%
# Let us reconstruct the signal from this sparse representation
y = prob.reconstruct(x)
# %%
# The estimated measurements
b = A.times(x)


# %%
# Let us visualize the original and reconstructed representation
plot_representations(x0, x)


# %%
# Let us visualize the original and reconstructed signal
plot_signals(y0, y)

# %%
# Let us visualize the original and reconstructed measurements
plot_measurments(b0, b)

# %% 
# Comments
# ---------
# 
# * We need 115 coefficients in the representation in the
#   Daubechies basis to cover 99.9% of the signal energy.
#   There are a total of 278 nonzero coefficients. 
# 
# Subspace Pursuit
# 
# * With K=100 (< 115), Subspace Pursuit recovery is not very good (low SNR).
#   It converges in 20 iterations.
# * With K=150, SP is pretty good.
#   All detected nonzero coefficients are part of the true support.
# * With K=200, the SNR further improves to 54 dB.
# * With K=250, the SNR further improves to 83 dB.
#   All detected coefficients are correct so far.
# * Pushing the sparsity to K=278 stars causing problems however.
#   We can see the SNR drop to 76 dB.
#   We can notice that only 243 of the detected 277 coefficients
#   are correct coefficients.
#
# SPGL1
#
# * We use the basis pursuit with inequality constraints version of
#   SPGL1 in this example.
# * The allowed sigma for the residual norm :math:`\| \bA \bx - \bb \|_2`
#   is chosen as a fraction of the norm of the measurements :math:`\| \bb \|_2`.
# * At a fraction of 0.01, SPGL1 converges in 106 iterations giving an
#   SNR of 27 dB.
# * At a fraction of 0.001, SPGL1 converges in 486 iterations
#   with an improved SNR of 29 dB.
# * It is interesting to note that while the measurement SNR has improved
#   remarkably from 40 dB to 60 dB (as the target residual norm has reduced
#   by a factor of 10), the improvement in signal SNR is not that good.
#   Having a tighter bound on residual norm doesn't lead significantly better
#   reconstruction.

