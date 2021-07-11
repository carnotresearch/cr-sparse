Linear Operators
=======================


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


Signal processing operators
------------------------------------

.. autosummary::
    :toctree: _autosummary

    fourier_basis_1d


Convenience operators
-----------------------

These operators are technically not linear on :math:`\mathbb{F}^n \to \mathbb{F}^m`

.. autosummary::
    :toctree: _autosummary

    real

Operator algebra
------------------

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


Utilities
-------------------

.. autosummary::
    :toctree: _autosummary

    to_matrix
    to_adjoint_matrix
