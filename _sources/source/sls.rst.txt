Sparse Linear Systems
=========================

The solvers in this module focus on 
traditional least square problems
for square or overdetermined linear systems
:math:`A x = b` 
where the matrix :math:`A` is sparse and is 
represented by a linear operator abstraction 
providing the matrix multiplication and adjoint 
multiplication functions.

.. currentmodule:: cr.sparse.sls

Solvers
-----------------

.. autosummary::
    :toctree: _autosummary

    lsqr
    lsqr_jit
    power_iterations
    power_iterations_jit
    ista
    ista_jit
    fista
    fista_jit

Data types
------------------


.. autosummary::
    :toctree: _autosummary
    :nosignatures:
    :template: namedtuple.rst

    LSQRSolution
    PowerIterSolution
    ISTAState
    FISTAState


