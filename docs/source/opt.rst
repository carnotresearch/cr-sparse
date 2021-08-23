Numerical Optimization Routines
==================================


This section includes several routines which 
form basic building blocks for other higher level
solvers.

.. currentmodule:: cr.sparse.opt


Projections
--------------

.. autosummary::
  :toctree: _autosummary

  project_to_ball
  project_to_box
  project_to_real_upper_limit



Shrinkage
---------------------

.. autosummary::
  :toctree: _autosummary

  shrink


Conjugate Gradient Methods
----------------------------------

.. rubric:: Normal Conjugate Gradients on Matrices

.. autosummary::
  :toctree: _autosummary

    cg.solve_from
    cg.solve_from_jit
    cg.solve
    cg.solve_jit


.. rubric:: Preconditioned Normal Conjugate Gradients on Linear Operators

These are more general purpose.

.. autosummary::
  :toctree: _autosummary

    pcg.solve_from
    pcg.solve_from_jit
    pcg.solve
    pcg.solve_jit

