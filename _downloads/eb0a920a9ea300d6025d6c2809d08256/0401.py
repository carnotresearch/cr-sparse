r"""
.. _gallery:0401:

Source Separation I
=======================

.. contents::
    :depth: 2
    :local:


In audio signal processing,
source separation is the process of isolating individual sounds
in an auditory mixture of multiple sounds.

Modeling the Source Separation Problem
-------------------------------------------

In this example we consider the problem
of separating two sources which have been
mixed together. 
One is the sound of a guitar,
and another is the sound of piano.
The mixture is being sensor from two
different places with different mixing
weights at each sensor. For each time
instant, the mixing process can be
described by the equation:

.. math::

    \bv = \bM \bu

where :math:`\bu` is a 2-vector containing
one sample each from the guitar and piano
sounds and :math:`\bv` is a 2-vector containing
the samples from the two different listening sensors.
The matrix :math:`\bM` is a :math:`2 \times 2`
mixing matrix.
The mixing matrix
is known to us in this problem.

Assume that the audio has been captured for :math:`m`
samples. We can put together the source audio in
a :math:`m \times 2` matrix :math:`\by` and the captured
audio from the two sensors as a :math:`m \times 2`
matrix :math:`\bb` with the mixing relationship given by

.. math::

    \bb = \by \bM^T

Here the post multiplication by the matrix
:math:`\bM^T` is equality to processing each
row of input :math:`\by` by the mixing process
to generate a row of output.

The Sensor
''''''''''''''''''

From the perspective of sparse reconstruction,
we treat the mixing process as compressive sampling.
Then we can write the mixing equation as the application
of a linear operator :math:`\Phi`:

.. math::

    \bb = \Phi (\by).

Beware that this linear operator processes the input
:math:`\by` row by row to generate the output :math:`\bb`.
Fortunately, the CR-Sparse linear operator architecture allows
us to process input data along any axis of choice. By default
all linear operators process data column by column, which
is akin to a matrix pre-multiplication. However, in this
problem, we will use a simple :math:`2 \times 2` matrix
post-multiplication as our linear operator for representing
the sensing process. 

An alternative design (as used in SPARCO) would have been
to flatten :math:`\by` to a vector of length :math:`2 m`
and then use a :math:`2 m \times 2 m` sensing matrix :math:`\Phi`
(obtained by computing the Kronecker product of :math:`\bM`
with an identity matrix :math:`\bI_m`) to construct a flattened
version of :math:`\bb`. Our post-multiplication version of
linear operator is a more efficient implementation.


The Sparsifying Basis
''''''''''''''''''''''''

To create a suitable sparsifying basis for this signal,
we will first consider a forward transform as defined below:

#. Split the signal into overlapping windows of length :math:`w` 
   samples each.
#. Let the overlap between windows be of :math:`l = w/2`
   samples (50%) overlap.
#. For the last partial window, pad it with zeros as needed.
#. Compute the DCT transform of each window.
#. Concatenate the transforms together to form a representation
   of the signal.

Let there be :math:`b` such
overlapping windows over a signal of length :math:`m`. Then
the shape of this forward transform operator is :math:`w b \times m`.

We define the adjoint (transpose) of the above forward transform as
the Windowed Discrete Cosine Basis for our music sounds. We denote
this basis by :math:`\Psi`. The representation equation is given by

.. math::

    \by = \Psi (\bx).

Here :math:`\bx` is an :math:`w b \times 2` representation matrix
where each column is a representation of each column in :math:`\by`.
In other words, the first column of :math:`\bx` is a :math:`wb` length
representation of the guitar signal and the second column is the
representation of the piano signal. This is an overcomplete
basis (processing input data column by column). Note that by design,
each column of :math:`\Psi` is unit length.

CR-Sparse includes an operator called ``windowed_op`` which can
transform any linear operator :math:`T` into a windowed operator
as per description above. Starting from the ``cosine_basis``
operator, we construct our sparsifying basis :math:`\Psi`
using ``windowed_op``. In particular we will be using
512 length windows with overlaps of 256 samples.
The details of the construction of the sensing operator
and the sparsifying basis can be seen in the ``prob401.py``
file in the source code.

The Sparse Recovery Problem
'''''''''''''''''''''''''''''

We now combine the operators :math:`\Phi` and :math:`\Psi`
to construct the operator :math:`\bA = \Phi \Psi` and form
the linear equation:

.. math::

    \bb = \bA (\bx).

We can now use a suitable sparse recovery algorithm to
recover :math:`\bx` from :math:`\bb`. We shall do
that using SPGL1 (Spectral Gradient Descent for L1)
in this example.

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
prob = problems.generate('src-sep-1')
fig, ax = problems.plot(prob)

# %% 
# Let us access the relevant parts of our test problem

# The sparsifying basis linear operator
Psi = prob.Psi
# The combined operator for the linear equation :math:`\bb = \bA \bx`
A = prob.A
# Mixture signals
b0 = prob.b
# Original signals
y0 = prob.y
# The sparse representation of the signal in the dictionary
x0 = prob.x

# %% 
# Sparse Reconstruction using SPGL1
# -------------------------------------


# %% 
# The shape of mixed signal 
bm, nc = b0.shape
# The shape of sparsifying basis
tm, tn = prob.Psi.shape
print(bm, nc, tm, tn)
# Prepare an initial (zero) estimate of the model
x_init = jnp.zeros((tn, nc))

# %% 
# Run SPGL1 algorithm
import cr.sparse.cvx.spgl1 as crspgl1
sigma=0.
options = crspgl1.SPGL1Options(max_iters=300)
sol = crspgl1.solve_bpic_from_jit(A, b0, sigma, 
    x_init, options=options)
problems.analyze_solution(prob, sol)

# %% 
# The estimated sparse representation
x = sol.x
# %%
# Let us reconstruct the signal from this sparse representation
y = prob.reconstruct(x)


ax = crplot.h_plots(4)
ax[0].plot(y0[:, 0])
ax[0].set_title("Original audio 1 (Guitar)")
ax[1].plot(y[:, 0])
ax[1].set_title("Reconstructed audio 1 (Guitar)")
ax[2].plot(y0[:, 1])
ax[2].set_title("Original audio 2 (Piano)")
ax[3].plot(y[:, 1])
ax[3].set_title("Reconstructed audio 2 (Piano)")
