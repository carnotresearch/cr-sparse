L1 Minimization
=============================================================================

.. contents::
    :depth: 2
    :local:

Truncated Newton Interior Points Method (TNIPM)
---------------------------------------------------------

This method can be used to solve problems of type:

.. math::

  \min \| A x  - b \|_2^2 + \lambda \| x \|_1 

The method works as follows:

* The :math:`\ell_1` regularized LSP (Least Squares Program) above is 
  transformed into a convex quadratic program with linear inequality constraints.
* The logarithmic barrier for the quadratic program is constructed.
* The central path for this logarithmic barrier minimizer is identified.
* Truncated newton method is used to solve for points on the central path.

  * Compute the search direction as an approximate solution to the Newton system
  * Compute the step size by backtracking line search
  * Update the iterate
  * Construct a dual feasible point 
  * Evaluate the duality gap
  * Repeat till the relative duality gap is within a specified threshold

* The Newton system is solved approximately using preconditioned conjugate gradients.

The ``solve_from`` versions below are useful if the solution is known partially.
The ``solve`` versions are more directly applicable when solution is not known.
Both ``solve_from`` and ``solve`` are available as regular and JIT-compiled 
versions.


.. currentmodule:: cr.sparse.cvx

.. autosummary::
  :toctree: _autosummary

    l1ls.solve_from
    l1ls.solve_from_jit
    l1ls.solve
    l1ls.solve_jit

An example is provided in :ref:`recovery_l1_gallery`.

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
