r"""
.. _gallery:0005:

Cosine+Spikes, Dirac-Cosine Basis, Gaussian Measurements
=============================================================

.. contents::
    :depth: 2
    :local:

In this example we have

#. A signal :math:`\by` consisting of a mixture of
   3 cosine waves and 60 random spikes of total length 1024.
#. A Dirac-Cosine two ortho basis :math:`\Psi` of shape 1024x2048.
#. The sparse representation :math:`\bx` of the signal :math:`\by`
   in the basis :math:`\Psi` consisting of exactly 63 nonzero
   entries (corresponding to the spikes and the amplitudes of the cosine waves).
#. A Gaussian sensing matrix :math:`\Phi` of shape  300x1024 making
   300 random measurements in a vector :math:`\bb`.
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
import cr.sparse.plots as crplot


# %% 
# Setup
# ------------------------------
# We shall construct our test signal and dictionary
# using our test problems module.

from cr.sparse import problems
prob = problems.generate('cosine-spikes:dirac-dct:gaussian', c=3, k=60)
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

# We will first try to estimate a 50-sparse representation
sol = sp.solve(A, b0, 50)
# %%
# This utility function helps us quickly analyze the quality of reconstruction
problems.analyze_solution(prob, sol)

# %%
# We will now try to estimate a 75-sparse representation
sol = sp.solve(A, b0, 75)
problems.analyze_solution(prob, sol)
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
# Sparse Recovery using Compressive Sampling Matching Pursuit
# ---------------------------------------------------------------
# We shall now use compressive sampling matching pursuit to reconstruct the signal.
import cr.sparse.pursuit.cosamp as cosamp
# We will try to estimate a 75-sparse representation
sol = cosamp.solve(A, b0, 75)
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
# Sparse Recovery using SPGL1
# ---------------------------------------------------------------
import cr.sparse.cvx.spgl1 as crspgl1
options = crspgl1.SPGL1Options(max_iters=1000)
sol = crspgl1.solve_bp_jit(A, b0, options=options)
problems.analyze_solution(prob, sol)

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
# * With K=50, SP recovery is slightly inaccurate.
#   It also takes more (20) iterations to converge.
# * With K=75, SP is pretty good. It only missed one of the
#   63 nonzero entries. Also, SP converges in just 10 iterations.
# * With K=75, CoSaMP is also good but slightly poor. It also
#   missed just one nonzero entry. But it seems like it missed
#   a more significant entry compared to SP. Also, CoSaMP took
#   57 iterations to converge.
# * SPGL1 converges in 788 iterations to converge.
#   Its model space and signal space SNR are not good.
#   However, its measurement space SNR is pretty high.

