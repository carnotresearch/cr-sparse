r"""
.. _gallery:0002:

Blocks Signal in Haar Basis
=============================

.. contents::
    :depth: 2
    :local:

A Blocks signal as proposed by Donoho et al.
in Wavelab :cite:`buckheit1995wavelab` is
a concatenation of blocks with different heights.
It has a sparse representation in a Haar basis.

In this test problem with the structure
:math:`\bb = \bA \bx`, the signal
:math:`\bb` is the blocks signal
(of length 1024), :math:`\bA`
is a Haar basis with
5 levels of decomposition and :math:`\bx`
is the sparse representation of the signal
in this basis.
This basis is real, complete and orthonormal.
Hence, a simple solution for getting
:math:`\bx` from :math:`\bb` is the formula:

.. math::

    \bx = \bA^T \bb.

This test problem is useful in identifying
basic mistakes in a sparse recovery algorithm.
This problem should be very easy to solve by
any sparse recovery algorithm. However, there
is a caveat. If a sparse recovery algorithm
depends on the expected sparsity of the
signal (typically a parameter K in greedy algorithms),
the reconstruction would fail if K is specified below
the actual number of significant components of :math:`\bx`.

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
prob = problems.generate('blocks:haar')
fig, ax = problems.plot(prob)


# %% 
# Let us access the relevant parts of our test problem

# The sparsifying basis linear operator
A = prob.A
# The Blocks signal
b0 = prob.b
# The sparse representation of the Blocks signal in the dictionary
x0 = prob.x


# %% 
# Check how many coefficients in the sparse representation
# are sufficient to capture 99.9% of the energy of the signal
print(crn.num_largest_coeffs_for_energy_percent(x0, 99.9))

# %% 
# This number gives us an idea about the required sparsity
# to be configured for greedy pursuit algorithms.

# %% 
# Sparse Recovery using Subspace Pursuit
# -------------------------------------------
# We shall use subspace pursuit to reconstruct the signal.
import cr.sparse.pursuit.sp as sp
# We will try to estimate a 100-sparse representation
sol = sp.solve(A, b0, 100)
# %%
# This utility function helps us quickly analyze the quality of reconstruction
problems.analyze_solution(prob, sol)

# %% 
# The estimated sparse representation
x = sol.x
# %%
# Let us reconstruct the signal from this sparse representation
b = prob.reconstruct(x)

# %%
# Let us visualize the original and reconstructed representation
ax = crplot.h_plots(2)
ax[0].stem(x0, markerfmt='.')
ax[1].stem(x, markerfmt='.')



# %%
# Let us visualize the original and reconstructed signal
ax = crplot.h_plots(2)
ax[0].plot(b0)
ax[1].plot(b)



# %% 
# Sparse Recovery using Compressive Sampling Matching Pursuit
# ---------------------------------------------------------------
# We shall now use compressive sampling matching pursuit to reconstruct the signal.
import cr.sparse.pursuit.cosamp as cosamp
# We will try to estimate a 100-sparse representation
sol = cosamp.solve(A, b0, 100)
problems.analyze_solution(prob, sol)

# %% 
# The estimated sparse representation
x = sol.x
# %%
# Let us reconstruct the signal from this sparse representation
b = prob.reconstruct(x)


# %%
# Let us visualize the original and reconstructed representation
ax = crplot.h_plots(2)
ax[0].stem(x0, markerfmt='.')
ax[1].stem(x, markerfmt='.')


# %%
# Let us visualize the original and reconstructed signal
ax = crplot.h_plots(2)
ax[0].plot(b0)
ax[1].plot(b)


# %% 
# Sparse Recovery using SPGL1
# ---------------------------------------------------------------
import cr.sparse.cvx.spgl1 as crspgl1
options = crspgl1.SPGL1Options()
sol = crspgl1.solve_bp_jit(A, b0, options=options)
problems.analyze_solution(prob, sol)

# %% 
# The estimated sparse representation
x = sol.x
# %%
# Let us reconstruct the signal from this sparse representation
b = prob.reconstruct(x)


# %%
# Let us visualize the original and reconstructed representation
ax = crplot.h_plots(2)
ax[0].stem(x0, markerfmt='.')
ax[1].stem(x, markerfmt='.')


# %%
# Let us visualize the original and reconstructed signal
ax = crplot.h_plots(2)
ax[0].plot(b0)
ax[1].plot(b)


# %% 
# Comments
# ---------
# We note that SP converges in a single iteration,
# CoSaMP takes two iterations, 
# while SPGL1 takes 9 iterations to converge.