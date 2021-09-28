.. _api:lop:

Linear Operators
=======================

.. contents::
    :depth: 2
    :local:

We provide a collection of linear operators with efficient JAX based implementations 
that are
relevant in standard signal/image processing problems.
We also provide a bunch of utilities to combine and convert
linear operators.

This module is inspired by ``pylops`` although the implementation approach 
is different.

A linear operator :math:`T : X \to Y` connects a model space :math:`X` 
to a data space :math:`Y`.

A linear operator satisfies following laws:

.. math::

    T (x + y) = T (x) + T (y)


.. math::

    T (\alpha x) = \alpha T(x)


Thus, for a general linear combination:

.. math::

    T (\alpha x + \beta y) = \alpha T (x) + \beta T (y)

We are concerned with linear operators :math:`T : \mathbb{F}^n \to \mathbb{F}^m`
where :math:`\mathbb{F}` is either the field of real numbers or 
complex numbers. 
:math:`X = \mathbb{F}^n` is the model space and 
:math:`Y = \mathbb{F}^m` is the data space.

Such a linear operator can be represented by a two dimensional matrix :math:`A`.

The forward operation is given by:

.. math::

    y = A x.

The corresponding adjoint operation is given by:

.. math::

    \hat{x} = A^H y

We represent a linear operator by a pair of functions ``times`` and ``trans``. 
The ``times`` function implements the forward operation while the ``trans``
function implements the adjoint operation.

An inverse problem consists of computing :math:`x` given :math:`y` and :math:`A`.

..  rubric:: 1D, 2D, ND operators

* A 1D operator takes a 1D array as input and returns a 1D array as output.
  E.g. identity, pad, etc.
* A 2D operator takes a 2D array as input and returns a 2D array as output. 
  E.g. conv2D, dwt2D, etc.
* An ND operator takes an ND array as input and returns an ND array as output.

The vectors may be stored using multi-dimensional arrays in memory. 
E.g., images are usually stored in 2-3 dimensional arrays.
The operators themselves may work directly on multi-dimensions arrays. 
E.g. a 2D convolution operator can be applied directly
to an image to result in another image. 

In other words, the vectors from the model space as well as data space 
may be stored in memory using 1D,2D,...,ND array.  They should still
be treated as vectors for the purposes of this module.

This is a departure from pylops convention where input to a 
linear operator must be flattened into a 1D array and output 
needs to be reshaped again.
In this library, the input and output to a 2D linear operator 
would be a 2D array.

.. rubric:: axis parameter in a 1D linear operator

* A 1D linear operator may get an ND array as input.
* In this case, the axis parameter to the operator specifies the
 axis along which the linear operator is to be applied.
* The input ND array will be broken into slices of 1D arrays along the
  axis and the linear operator will be applied separately to each slice.
* Then the slices will be combined to generate the output ND array.
* E.g. if the input is a matrix then:
  
  * axis=0 means apply the linear operator over each column (along axis=0)
  * axis=1 means apply the linear operator over each row  (along axis=1)

This is based on the convention followed by ``numpy.apply_along_axis``.

.. currentmodule:: cr.sparse.lop


Data types
------------------


.. autosummary::
    :toctree: _autosummary
    :nosignatures:
    :template: namedtuple.rst

    Operator

Basic operators
------------------

.. autosummary::
    :toctree: _autosummary

    identity
    matrix
    diagonal
    zero
    flipud
    sum
    pad_zeros
    symmetrize
    restriction

Operator algebra
------------------

It is possible to combine one or more linear operators
to create new linear operators. The functions in this
section provide different ways to combine linear operators.

.. autosummary::
    :toctree: _autosummary

    neg
    scale
    add
    subtract
    compose
    transpose
    hermitian
    hcat
    power
    block_diag

Signal processing operators
------------------------------------

.. autosummary::
    :toctree: _autosummary

    running_average
    fir_filter
    convolve
    convolve2D
    convolveND


Orthonormal transforms and bases
------------------------------------------------

.. autosummary::
    :toctree: _autosummary

    dwt
    dwt2D
    fourier_basis
    cosine_basis
    walsh_hadamard_basis

Unions of bases
--------------------

.. autosummary::
    :toctree: _autosummary

    dirac_fourier_basis


Random compressive sensing operators
--------------------------------------

.. autosummary::
    :toctree: _autosummary

    gaussian_dict
    rademacher_dict
    random_onb_dict
    random_orthonormal_rows_dict


Operators for special matrices
------------------------------------

.. autosummary::
    :toctree: _autosummary

    circulant


Derivatives (finite differences)
--------------------------------------

.. autosummary::
    :toctree: _autosummary

    first_derivative
    second_derivative

Convenience operators
-----------------------

These operators are technically not linear on :math:`\mathbb{F}^n \to \mathbb{F}^m`

.. autosummary::
    :toctree: _autosummary

    real


Operator parts
------------------

.. autosummary::
    :toctree: _autosummary

    column
    columns

Properties of a linear operator
--------------------------------------------

These are still experimental and not efficient.

.. autosummary::
    :toctree: _autosummary

    upper_frame_bound


Utilities
-------------------

.. autosummary::
    :toctree: _autosummary

    jit
    to_matrix
    to_adjoint_matrix
    to_complex_matrix
