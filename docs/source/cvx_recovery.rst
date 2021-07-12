Convex Optimization based Sparse Recovery/Approximation Algorithms
=============================================================================

.. contents::
    :depth: 2
    :local:

Alternating Directions Methods
-------------------------------------

A tutorial has been provided to explore these 
methods in action. 
The ``yall1.solve`` method is an overall wrapper method 
for solving different types of :math:`\ell_1` minimization
problems. It in turn calls the lower level methods for solving
specific types of problems.

.. currentmodule:: cr.sparse.cvx.adm

.. autosummary::
  :toctree: _autosummary

    yall1.solve
    yall1.solve_bp
    yall1.solve_bp_jit
    yall1.solve_l1_l2
    yall1.solve_l1_l2_jit
    yall1.solve_l1_l2con
    yall1.solve_l1_l2con_jit
