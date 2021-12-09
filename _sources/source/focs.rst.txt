First Order Conic Solvers
=================================

.. contents::
    :depth: 2
    :local:

.. currentmodule:: cr.sparse.focs

This module aims to implement the first order conic solvers 
for sparse signal recovery problems proposed in :cite:`becker2011templates`.
The implementation is adapted from TFOCS :cite:`becker2012tfocs`.

We consider problems of the form:

.. math::

    \text{minimize } \phi(x) = f( \AAA(x) + b) + h(x)

where:

* :math:`\AAA` is a linear operator from :math:`\RR^n \to \RR^m`.
* :math:`b` is a translation vector.
* :math:`f : \RR^m \to \RR` is a *smooth* convex function.
* :math:`h : \RR^n \to \RR` is a *prox-capable* convex function.

For a smooth function, its gradient :math:`g = \nabla f` must exist and be
easy to compute.

For a prox-capable function, it should have an efficient proximal operator:

.. math::

    p_f(x, t) = \text{arg} \min_{z \in \RR^n} f(x) + \frac{1}{2t} \| z - x \|_2^2

for any :math:`x \in \RR^n` and the step size :math:`t > 0`.

See the following sections for details:

* :ref:`api:lop`
* :ref:`api:opt:smooth`
* :ref:`api:opt:proximal`


The routine :py:func:`focs` provides the solver for the minimization
problem described above.

* Unconstrained smooth minimization problems can be handled by choosing
  :math:`h(x) = 0`. See :py:func:`cr.sparse.opt.prox_zero`.
* Convex constraints can be handled by adding their indicator functions 
  as part of :math:`h`.

For solving the minimization problem, an initial solution :math:`x_0` 
should be provided. If one is unsure of the initial solution, they can
provide :math:`0 \in \RR^n` as the initial solution.




Solvers
------------------

.. autosummary::
    :toctree: _autosummary

    focs
    l1rls
    l1rls_jit
    lasso


Data types
------------------


.. autosummary::
    :toctree: _autosummary
    :nosignatures:
    :template: namedtuple.rst

    FOCSOptions
    FOCSState


Utilities
------------------

.. autosummary::
    :toctree: _autosummary

    matrix_affine_func



In the rest of the document, we will discuss how specific 
sparse signal recovery problems can be modeled and solved
using this module.

L1 regularized least square problem
-----------------------------------------


We consider the problem:

.. math::

    \text{minimize} \frac{1}{2} \| \AAA x - b \|_2^2 + \lambda \| x \|_1 


Choose:

- :math:`f(x) = \frac{1}{2}\| x \|_2^2`  see :func:`cr.sparse.opt.smooth_quad_matrix`
- :math:`h(x) = \| \lambda x \|_1` see :func:`cr.sparse.opt.prox_l1`
- :math:`\AAA` as the linear operator
- :math:`-b` as the translate input

With these choices, it is straight-forward to use :func:`focs` to solve 
the L1LRS problem.  This is implemented in the function :func:`l1rls`.


LASSO
------------------

LASSO (least absolute shrinkage and selection operator) 
s a regression analysis method that performs both variable selection 
and regularization in order to enhance the prediction accuracy 
and interpretability of the resulting statistical model.

We consider the problem: 

.. math::

    \begin{aligned}
    \underset{x}{\text{minimize}} \frac{1}{2} \| \AAA x - b \|_2^2\\
    \text{subject to } \| x \|_1 \leq \tau
    \end{aligned}


Choose:

- :math:`f(x) = \frac{1}{2}\| x \|_2^2`  see :func:`cr.sparse.opt.smooth_quad_matrix`
- :math:`h(x) = I_C(x)` as the indicator function for l1-ball :math:`C = \{x : \| x \|_1 \leq \tau\}`, 
  see :func:`cr.sparse.opt.prox_l1_ball`
- :math:`\AAA` as the linear operator
- :math:`-b` as the translate input

With these choices, it is straight-forward to use :func:`focs` to solve 
the LASSO problem.  This is implemented in the function :func:`lasso`.
