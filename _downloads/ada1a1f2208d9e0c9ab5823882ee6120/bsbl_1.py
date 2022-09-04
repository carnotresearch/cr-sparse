r"""
.. _gallery:cs:bsbl:1:


Block Sparse Bayesian Learning
=================================

.. contents::
    :depth: 2
    :local:

In this example, we demonstrate the
BSBL (Block Sparse Bayesian Learning) algorithm
:cite:`zhang2012recovery,zhang2013extension`
for reconstruction of block sparse signals
with intra block correlations from their
compressive measurements. In particular,
we show

- Creation of block sparse signals with intra-block correlation
- Compressive sampling of the signal with Gaussian and sparse
  binary sensing matrices.
- Reconstruction using BSBL EM algorithm.


Our implementation of BSBL is fully JIT compilable.
To achieve this, we limit ourselves of equal sized
blocks where the block size is user defined. This
is not a problem in practice. As shown in
:cite:`zhang2012compressed`, the reconstruction
from compressive measurements of real life signals
is not affected much by the block size. 


The basic compressive sensing model is given by

.. math::

    \by = \Phi \bx + \be

where :math:`\by` is a known measurement vector,
:math:`\Phi` is a known sensing matrix and
:math:`\bx` is a sparse signal to be recovered
from the measurements.

We introduce the block/group structure on :math:`\bx`
as

.. math::

    \bx = \begin{pmatrix}
    \bx_1 & \bx_2 & \dots & \bx_g
    \end{pmatrix}

where each :math:`\bx_i` is a block of :math:`b`
values. The signal :math:`\bx` consists of :math:`g`
such blocks/groups.  We only consider the case of
equal sized blocks in our implementation.
Under the block sparsity model, only a few
:math:`k \ll g` blocks are nonzero (active)
in the signal :math:`\bx` however, the locations
of these blocks are unknown.

We can rewrite the sensing equation as:

.. math::

    \by = \sum_{i=1}^g \Phi_i \bx_i + \be

by splitting the sensing matrix into blocks of columns appropriately.

Under the sparse Bayesian framework, each block
is assumed to satisfy a parametrized multivariate
Gaussian distribution:

.. math::

    \PP(\bx_i ; \gamma_i, \bB_i) = \NNN(\bzero, \gamma_i \bB_i), \Forall i=1,\dots,g.

The covariance matrix :math:`\bB_i` captures the intra block correlations.



We further assume that the blocks are mutually uncorrelated.
The prior of :math:`\bx` can then be written as

.. math::

    \PP(\bx; \{ \gamma_i, \bB_i\}_i ) = \NNN(\bzero, \Sigma_0)

where

.. math::

    \Sigma_0 = \text{diag}\{\gamma_1 \bB_1, \dots, \gamma_g \bB_g \}.


We also model the correlation among the values
within each active block as an AR-1 process. Under this
assumption the matrix :math:`\bB_i` take the form of a Toeplitz
matrix

.. math::

    \bB = \begin{bmatrix}
    1 & r & \dots & r^{b-1}\\
    r & 1 & \dots & r^{b-2}\\
    \vdots &  & \ddots & \vdots\\
    r^{b-1} & r^{b-2} & \dots & 1
    \end{bmatrix}

where :math:`r` is the AR-1 model coefficient. This constraint
significantly reduces the model parameters to be learned.

Measurement noise is modeled as independent zero mean Gaussian
noise :math:`\PP(\be; \lambda) \sim \NNN(\bzero, \lambda \bI)`.
BSBL doesn't require you to provide the value of noise variance
as input. It is able to estimate :math:`\lambda` within a algorithm.

The estimate of :math:`\bx` under Bayesian learning framework
is given by the posterior mean of :math:`\bx` given the measurements
:math:`\by`.


Please also refer to the
`BSBL website <http://dsp.ucsd.edu/~zhilin/BSBL.html>`_
by the authors of the original algorithm for further information.

Related Examples

- :ref:`gallery:cs:sparse_binary_sensor`
"""

# %% 
# Setup
# ------------------------------

# Configure JAX for 64-bit computing
from jax.config import config
config.update("jax_enable_x64", True)

# %% 
# Let's import necessary libraries

from matplotlib import pyplot as plt
# jax imports
import jax.numpy as jnp
from jax import random, jit
# cr-suite imports
import cr.nimble as crn
import cr.sparse as crs
import cr.sparse.dict as crdict
import cr.sparse.data as crdata
import cr.sparse.plots as crplot

import cr.sparse.block.bsbl as bsbl


# %% 
# Problem Configuration
# ------------------------------

# ambient dimension
n = 300
# block length
b = 4
# number of blocks
nb = n // b
# Block sparsity: number of nonzero blocks
k = 6
# Number of measurements
m = 100

# %% 
# Block Sparse Signal
# ------------------------------

# Block sparse signal with intra block correlation
x, blocks, indices  = crdata.sparse_normal_blocks(
    crn.KEYS[2], n, k, b, cor=0.9, normalize_blocks=True)
ax = crplot.one_plot()
ax.stem(x);


# %% 
# Gaussian Sensing
# ------------------------------

# %% 
# Sensing matrix
Phi = crdict.gaussian_mtx(crn.KEYS[0], m, n, normalize_atoms=True)
ax = crplot.one_plot()
ax.imshow(Phi);

# %% 
# Measurements
y = Phi @ x
ax = crplot.one_plot()
crplot.plot_signal(ax, y)

# %% 
# Reconstruction using BSBL EM
# ------------------------------
# We need to provide the sensing matrix, measurements
# and the block size as parameters to the
# reconstruction algorithm
sol = bsbl.bsbl_em_jit(Phi, y, b)
print(sol)

# %% 
# Reconstructed signal
x_hat = sol.x
print(f'PRD: {crn.prd(x, x_hat):.1f} %, SNR: {crn.signal_noise_ratio(x, x_hat)} dB.' )


# %% 
# Plot the original and reconstructed signal
ax = crplot.h_plots(2)
ax[0].stem(x)
ax[1].stem(x_hat)


# %% 
# Sparse Binary Sensing
# ------------------------------

# %%
# We shall have just 12 ones in each column of the sensing matrix
d = 12
# %% 
# Build the sensing matrix
Phi = crdict.sparse_binary_mtx(crn.KEYS[0], m, n, d, 
    normalize_atoms=True, dense=True)
ax = crplot.one_plot()
ax.spy(Phi);
# %% 
# Measurements
y = Phi @ x
ax = crplot.one_plot()
crplot.plot_signal(ax, y)

# %% 
# Reconstruction using BSBL EM
# ------------------------------
# We need to provide the sensing matrix, measurements
# and the block size as parameters to the
# reconstruction algorithm
sol = bsbl.bsbl_em_jit(Phi, y, b)
print(sol)

# %% 
# Reconstructed signal
x_hat = sol.x
print(f'PRD: {crn.prd(x, x_hat):.1f} %, SNR: {crn.signal_noise_ratio(x, x_hat)} dB.' )


# %% 
# Plot the original and reconstructed signal
ax = crplot.h_plots(2)
ax[0].stem(x)
ax[1].stem(x_hat)
