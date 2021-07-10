Linear Operators
=======================


We provide a collection of linear operators which are
relevant in standard signal/image processing problems.
We also provide a bunch of utilities to combine and convert
linear operators.

.. currentmodule:: cr.sparse.lop


Data types
------------------


.. autosummary::
    :toctree: _autosummary

    LinearOperator

Basic operators
------------------

.. autosummary::
    :toctree: _autosummary

    identity
    matrix
    diagonal


Operator algebra
------------------

.. autosummary::
    :toctree: _autosummary

    neg
    scale
    add
    subtract
    compose
    hcat
    power


Utilities
-------------------

.. autosummary::
    :toctree: _autosummary

    to_matrix
