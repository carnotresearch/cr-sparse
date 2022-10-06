.. _api:l1min:

Convex Relaxation
=============================================================================

.. contents::
    :depth: 2
    :local:

.. _api:l1min:tnipm:

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


.. _api:l1min:admmm:

Alternating Directions Methods
-------------------------------------

This is based on :cite:`yang2011alternating`.

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


FOcal Underdetermined System Solver (FOCUSS)
-----------------------------------------------

This is based on :cite:`elad2010sparse` (section 3.2.1).

.. currentmodule:: cr.sparse.cvx

.. autosummary::
    :toctree: _autosummary

    focuss.matrix_solve_noiseless
    focuss.step_noiseless

Example:

* :ref:`gallery:focuss:1`



Spectral Projected Gradient L1 (SPGL1)
------------------------------------------

Berg and Friedlander proposed the spectral
projected gradient l1 (SPGL1) algorithm
in :cite:`BergFriedlander:2008`.
They provide a MATLAB package implementing
the algorithm in :cite:`spgl1site`.
A NumPy based Python port is also available
`here <https://pypi.org/project/spgl1/>`_.
We have implemented the algorithm with JAX.

.. autosummary::
    :toctree: _autosummary

    spgl1.solve_lasso_from
    spgl1.solve_lasso
    spgl1.solve_lasso_jit
    spgl1.solve_bpic_from
    spgl1.solve_bpic_from_jit
    spgl1.solve_bpic
    spgl1.solve_bpic_jit
    spgl1.solve_bp
    spgl1.solve_bp_jit

.. rubric:: Associated data types

.. autosummary::
    :toctree: _autosummary

    spgl1.SPGL1Options
    spgl1.SPGL1LassoState
    spgl1.SPGL1BPState

.. rubric:: Examples

* :ref:`gallery:0002`

Theory
''''''''''''


