r"""
.. _gallery:0001:

HeaviSine Signal in Dirac-Fourier Basis
===========================================

.. contents::
    :depth: 2
    :local:

A HeaviSine signal as proposed by Donoho et al.
in Wavelab :cite:`buckheit1995wavelab` is a sign
wave with jump discontinuities. It is not sparse
in standard (dirac) basis. It also doesn't have
a sparse representation in the Fourier basis.
However, it does have a sparse representation
in the Fourier-Heaviside dictionary.

In this example, we will generate a test problem
of HeaviSine signal and construct its sparse
representation through various algorithms.

The dictionary is a concatenation of the
(orthonormal) Fourier basis and a (non-orthogonal)
HeaviSide basis.

HeaviSide basis is derived from the
`HeaviSide step function <Heaviside step function>`_.
In its finite dimensional discrete form it looks
like an :math:`n \times n` matrix that has ones below
and on the diagonal and zeros elsewhere.
In other words, all the elements above the diagonal
are zero and rest are one.

Typical sparse reconstruction algorithms assume that
the atoms in a sparsifying dictionary have unit norms.
We provide both the unnormalized (with one in lower triangular
part) and normalized versions of the HeaviSide basis
in ``cr.sparse.lop`` module.
In this example, we shall use the normalized HeaviSide
basis.

Since the dictionary contains a Fourier basis, hence
the representation of the HeaviSine signal in this
dictionary is a complex valued representation.
The signal itself however is real.

See also:

* :ref:`api:problems`
* :ref:`api:lop`
* :ref:`api:l1min`
"""

# Configure JAX to work with 64-bit floating point precision. 
from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import cr.nimble as crn

# %% 
# Setup
# ------------------------------
# We shall construct our test signal and dictionary
# using our test problems module.

from cr.sparse import problems
prob = problems.generate('heavi-sine:fourier:heavi-side')
fig, ax = problems.plot(prob)

# %% 
# Let us access the relevant parts of our test problem

# The sparsifying basis linear operator
A = prob.A
# The HeaviSine signal
b0 = prob.b
# The sparse representation of the HeaviSide signal in the dictionary
x0 = prob.x


# %% 
# Check how many coefficients in the sparse representation
# are sufficient to capture 99.9% of the energy of the signal
print(crn.num_largest_coeffs_for_energy_percent(x0, 99.9))

# %% 
# Sparse Recovery using Subspace Pursuit
# -------------------------------------------
# We shall use subspace pursuit to reconstruct the signal.
import cr.sparse.pursuit.sp as sp
# We will try to estimate a 10-sparse representation
sol = sp.solve(prob.A, prob.b, 10)
print(sol)

# %% 
# The sparse representation estimated by subspace pursuit
x = sol.x
# %%
# Let us reconstruct the signal from this sparse representation
b = prob.reconstruct(x)

# %%
# Check if we could reconstruct the sparse representation correctly
print(f'Model Space SNR: {crn.signal_noise_ratio(x0, x)} dB')
# %%
# The SNR between the expected sparse representation
# and the recovered sparse representation is low.
# HeaviSide basis is highly correlated (high coherence).
# Hence, we
# couldn't make the exact recovery of the original sparse
# representation. Nevertheless we indeed recovered a good sparse
# representation since the residual norm is very small

# %%
# Check if we could reconstruct the signal correctly
print(f'Signal Space SNR: {crn.signal_noise_ratio(b0, b)} dB')
# %%
# The reconstruction is indeed excellent.



# %%
# Let us visualize the original and reconstructed signal
import cr.sparse.plots as crplot
ax = crplot.h_plots(2)
ax[0].plot(b0)
ax[1].plot(b)
