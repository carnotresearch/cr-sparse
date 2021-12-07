.. _api:opt:

Numerical Optimization Routines
==================================


This section includes several routines which 
form basic building blocks for other higher level
solvers.

.. currentmodule:: cr.sparse.opt

Indicator Function Generators
------------------------------

.. autosummary::
  :toctree: _autosummary
  
  indicator_zero
  indicator_singleton
  indicator_affine
  indicator_box
  indicator_box_affine
  indicator_conic
  indicator_l1_ball
  indicator_l2_ball


Projection Function Generators
---------------------------------

.. autosummary::
  :toctree: _autosummary


  proj_zero
  proj_identity
  proj_singleton
  proj_affine
  proj_box
  proj_conic
  proj_l1_ball
  proj_l2_ball



Proximal Operator Generators
----------------------------------

.. autosummary::
  :toctree: _autosummary

  prox_zero
  prox_l1
  prox_l2
  prox_l1_pos

Building proximal operators:

.. autosummary::
  :toctree: _autosummary

  ProxCapable
  prox_build





Smooth Function Generators
----------------------------------

.. autosummary::
  :toctree: _autosummary

  smooth_constant
  smooth_entropy
  smooth_huber
  smooth_linear
  smooth_entropy
  smooth_logdet
  smooth_quad_matrix

Operations on smooth functions:

.. autosummary::
  :toctree: _autosummary

  smooth_func_translate

Building smooth functions:

.. autosummary::
  :toctree: _autosummary

  SmoothFunction
  smooth_value_grad
  smooth_build
  smooth_build2
  smooth_build3

Simpler Projection Functions
----------------------------------

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

