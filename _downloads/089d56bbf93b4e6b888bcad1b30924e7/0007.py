r"""
.. _gallery:0007:

Signed Spikes, Gaussian Measurements
=============================================================

.. contents::
    :depth: 2
    :local:

In this example we have

#. A signal :math:`\bx` of length :math:`n=2560` with
   :math:`k=20` signed spikes.
   Each spike has a magnitude of 1.
   The sign for each spike is randomly assigned.
   The locations of spikes in the signal are also randomly chosen.
#. A Gaussian sensing matrix :math:`\Phi` of shape
   :math:`m \times n = 600 \times 2560` making
   600 random measurements in a vector :math:`\bb`
   given by the sensing equation :math:`\bb = \Phi \bx`.
   The columns of the sensing matrix are unit normalized.

The signal is sparse in standard basis.
This is a relatively easy sparse recovery problem and focuses on
the compressive sensing process modeled as:

.. math::

    \bb = \bA \bx = \Phi \bx.

See also:

* :ref:`api:problems`
* :ref:`api:lop`
"""


# Configure JAX to work with 64-bit floating point precision. 
from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import cr.nimble as crn
import cr.sparse.plots as crplot


# %% 
# Setup
# ------------------------------
# We shall construct our test signal, measurements and sensing matrix
# using our test problems module.

from cr.sparse import problems
k=20
m=600
n=2560
prob = problems.generate('signed-spikes:dirac:gaussian', k=k, m=m, n=n)
fig, ax = problems.plot(prob)


# %% 
# Let us access the relevant parts of our test problem

# The Gaussian sensing matrix operator
A = prob.A
# The measurements
b0 = prob.b
# The sparse signal
x0 = prob.x


# %% 
# Sparse Recovery using Subspace Pursuit
# -------------------------------------------
# We shall use subspace pursuit to reconstruct the signal.
import cr.sparse.pursuit.sp as sp
# We will try to estimate a k-sparse representation
sol = sp.solve(A, b0, k)
# %%
# This utility function helps us quickly analyze the quality of reconstruction
problems.analyze_solution(prob, sol)

# %% 
# The estimated sparse signal
x = sol.x
# %%
# Let us reconstruct the measurements from this signal
b = A.times(x)

# %%
# Let us visualize the original and reconstructed signal

def plot_signals(x0, x):
    ax = crplot.h_plots(2)
    ax[0].stem(x0, markerfmt='.')
    ax[0].set_title('Original signal')
    ax[1].stem(x, markerfmt='.')
    ax[1].set_title('Reconstructed signal')
plot_signals(x0, x)


# %%
# Let us visualize the original and reconstructed measurements
def plot_measurments(b0, b):
    ax = crplot.h_plots(2)
    ax[0].plot(b0)
    ax[0].set_title('Original measurements')
    ax[1].plot(b)
    ax[1].set_title('Reconstructed measurements')
plot_measurments(b0, b)

# %% 
# Sparse Recovery using Compressive Sampling Matching Pursuit
# ---------------------------------------------------------------
# We shall now use compressive sampling matching pursuit to reconstruct the signal.
import cr.sparse.pursuit.cosamp as cosamp
# We will try to estimate a k-sparse representation
sol = cosamp.solve(A, b0, k)
problems.analyze_solution(prob, sol)

# %% 
# The estimated sparse signal
x = sol.x
# %%
# Let us reconstruct the measurements from this signal
b = A.times(x)


# %%
# Let us visualize the original and reconstructed signals
plot_signals(x0, x)


# %%
# Let us visualize the original and reconstructed measurements
plot_measurments(b0, b)


# %% 
# Sparse Recovery using SPGL1
# ---------------------------------------------------------------
import cr.sparse.cvx.spgl1 as crspgl1
options = crspgl1.SPGL1Options()
sol = crspgl1.solve_bp_jit(A, b0, options=options)
problems.analyze_solution(prob, sol)

# %% 
# The estimated sparse signal
x = sol.x
# %%
# Let us reconstruct the measurements from this signal
b = A.times(x)


# %%
# Let us visualize the original and reconstructed signals
plot_signals(x0, x)


# %%
# Let us visualize the original and reconstructed measurements
plot_measurments(b0, b)
