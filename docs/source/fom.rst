First Order Methods
=================================

.. contents::
    :depth: 2
    :local:

.. currentmodule:: cr.sparse.fom

This module aims to implement the first order methods
for sparse signal recovery problems proposed in :cite:`becker2011templates`.
The implementation is adapted from TFOCS :cite:`becker2012tfocs`.

First order methods exploit information n values and gradients/subgradients 
(but not Hessians) of the functions comprising the models under consideration :cite:`beck2017first`.

API
---------------

.. rubric:: Solvers


.. autosummary::
    :toctree: _autosummary

    fom
    l1rls
    l1rls_jit
    lasso
    lasso_jit
    owl1rls
    owl1rls_jit


.. rubric:: Data types

.. autosummary::
    :toctree: _autosummary
    :nosignatures:
    :template: namedtuple.rst

    FomOptions
    FomState


.. rubric:: Utilities

.. autosummary::
    :toctree: _autosummary

    matrix_affine_func



First Order Methods
-----------------------

:cite:`becker2011templates` considers problems in the unconstrained form:

.. math::

    \text{minimize } \phi(x) = g( x) + h(x)

where:

* Both :math:`g, h: \RR^n \to (\RR \cup \infty)` are convex functions.
* :math:`g : \RR^m \to \RR` is a *smooth* convex function.
* :math:`h : \RR^n \to \RR` is a non-smooth convex function.

Of particular interest are problems where :math:`g` takes a specific form:

.. math::

  g(x) = f( \AAA(x) + b) 

where 

* :math:`\AAA` is a linear operator from :math:`\RR^n \to \RR^m`
* :math:`b` is a translation vector
* :math:`f : \RR^m \to \RR` is a *smooth* convex function

and the function :math:`h` is a *prox-capable* convex function (to be described later).


We can then rewrite :math:`\phi` as:

.. math::

    \text{minimize } \phi(x) = f( \AAA(x) + b) + h(x)

For a smooth function, its gradient :math:`\nabla f` must exist and be
easy to compute.

For a prox-capable function, it should have an efficient proximal operator:

.. math::

    p_f(x, t) = \underset{z \in \RR^n}{\text{arg} \min} \left ( f(x) + \frac{1}{2t} \| z - x \|_2^2 \right )

for any :math:`x \in \RR^n` and the step size :math:`t > 0`.

See the following sections for details:

* :ref:`api:lop`
* :ref:`api:opt:smooth`
* :ref:`api:opt:proximal`


The routine :py:func:`fom` provides the solver for the minimization
problem described above.

* Unconstrained smooth minimization problems can be handled by choosing
  :math:`h(x) = 0`. See :py:func:`cr.sparse.opt.prox_zero`.
* Convex constraints can be handled by adding their indicator functions 
  as part of :math:`h`.

For solving the minimization problem, an initial solution :math:`x_0` 
should be provided. If one is unsure of the initial solution, they can
provide :math:`0 \in \RR^n` as the initial solution.


Smooth Conic Dual Problems
---------------------------------

A number of sparse signal recovery problems can be cast as conic optimization problems. Then 
efficient first order methods can be used to solve the problem. This procedure often involves
the following steps:

- Cast the sparse recovery problem as a conic optimization problem
- Augment the objective function with a strongly convex term to make it smooth.
- Formulate the dual of the smooth problem.
- Solve the smooth conic dual problem using a first order method.
- Apply continuation to mitigate the effect of the strongly convex term added to the original objective function.


Our goal is to solve the conic problems of the form :cite:`becker2011templates`

.. math::

  \begin{aligned}
    & \underset{x}{\text{minimize}} 
    & &  f(x) \\
    & \text{subject to}
    & &  \AAA(x) + b \in \KKK
  \end{aligned}

where:

* :math:`x \in \RR^n` is the optimization variable 
* :math:`f :\RR^n \to \RR` is a convex objective function which is possibly
  extended valued but not necessarily smooth
* :math:`\AAA : \RR^n \to \RR^m` is a linear operator
* :math:`b \in \RR^m` is a translation vector 
* :math:`\KKK \subseteq \RR^m` is a closed, convex cone

.. rubric:: Dual Problem

The Lagrangian associated with the conic problem is given by:

.. math::

    \LLL(x, \lambda) = f(x)  - \langle \lambda, \AAA(x) + b \rangle 

where :math:`\lambda \in \RRR^m` is the Lagrange multiplier which must 
lie in the dual cone :math:`\KKK^*`.

The dual function is:

.. math::

    g(\lambda) = \underset{x \in \RR^n}{\inf} \LLL(x, \lambda)
    = - f^*(\AAA^*(\lambda)) - \langle b, \lambda \rangle 

* :math:`\AAA^*: \RR^m \to \RR^n` is the adjoint of the linear operator :math:`\AAA`
* :math:`f^* : \RR^n \to (\RR \cup \infty)` is the convex conjugate of :math:`f` defined by

.. math::

    f^*(z) = \underset{z}{\sup} \langle z, x \rangle -f(x)


Thus, the dual problem is given by:

.. math::

  \begin{aligned}
    & \underset{\lambda}{\text{maximize}} 
    & & - f^*(\AAA^*(\lambda)) - \langle b, \lambda \rangle \\
    & \text{subject to}
    & &  \lambda \in \KKK^*
  \end{aligned}


.. rubric:: Smoothing

The dual function :math:`g` may not be differentiable (or finite) on
all of :math:`\KKK^*`.

We perturb the original problem as follows:

.. math::

  \begin{aligned}
    & \underset{x}{\text{minimize}} 
    & &  f_{\mu} (x) \triangleq f(x) + \mu d (x)\\
    & \text{subject to}
    & &  \AAA(x) + b \in \KKK
  \end{aligned}

* :math:`\mu > 0` is a fixed smoothing parameter
* :math:`d : \RR^n \to \RR` is a strongly convex function obeying

.. math::

    d(x) \geq d(x_0) + \frac{1}{2} \| x - x_0 \|_2^2

A specific choice for :math:`d(x)` is:

.. math::

    d(x) = \frac{1}{2} \| x - x_0 \|_2^2

The new objective function :math:`f_{\mu}` is strongly convex.

The Lagrangian becomes: 

.. math::

    \LLL_{\mu}(x, \lambda) = f(x) + \mu d(x)  - \langle \lambda, \AAA(x) + b \rangle 


The dual function becomes: 


.. math::

    g_{\mu}(\lambda) = \underset{x \in \RR^n}{\inf} \LLL_{\mu}(x, \lambda)
    = - (f+\mu d)^*(\AAA^*(\lambda)) - \langle b, \lambda \rangle 

First order methods may be employed to solve this problem with 
provable performance. 




.. rubric:: Problem Instantiations

In the rest of the document, we will discuss how specific 
sparse signal recovery problems can be modeled and solved
using this module as either primal problems or a smooth conic dual problem.



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

With these choices, it is straight-forward to use :func:`fom` to solve 
the L1RLS problem.  This is implemented in the function :func:`l1rls`.


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

With these choices, it is straight-forward to use :func:`fom` to solve 
the LASSO problem.  This is implemented in the function :func:`lasso`.



Ordered weighted L1 regularized least square problem
-------------------------------------------------------------


We consider the problem:

.. math::

    \underset{x \in \RR^n}{\text{minimize}} \frac{1}{2} \| A x - b \|_2^2 + \sum_{i=1}^n \lambda_i | x |_{(i)} 

described in :cite:`lgorzata2013statistical`.


Choose:

- :math:`f(x) = \frac{1}{2}\| x \|_2^2`  see :func:`cr.sparse.opt.smooth_quad_matrix`
- :math:`h(x) = \sum_{i=1}^n \lambda_i | x |_{(i)}` see :func:`cr.sparse.opt.prox_owl1`
- :math:`\AAA` as the linear operator
- :math:`-b` as the translate input

With these choices, it is straight-forward to use :func:`fom` to solve 
the ordered weighted L1 regularized least square problem.  This is implemented in the function :func:`owl1rls`.


The Dantzig selector
----------------------------------

We wish to recover an unknown vector :math:`x_0 \in \RR^n` from the
data :math:`y \in \RR^m` and the model:

.. math::

    y = A x_0 + z 

where

* :math:`A` is a known :math:`(m, n)` sized design matrix
* :math:`z` is the noise term
* There are fewer observations/measurements than unknowns (:math:`m \lt n`)

We consider the optimization problem 

.. math::

  \begin{aligned}
    & \underset{x}{\text{minimize}} 
    & &  \| x \|_1 \\
    & \text{subject to}
    & &  \| A* (y - A x ) \|_{\infty} \leq \delta
  \end{aligned}

* We are minimizing the :math:`\ell_1` norm of the solution (thus promoting sparsity)
* We have an upper bound :math:`\delta` on the correlation between the 
  residual vector :math:`r = y - A x` and the columns/atoms of :math:`A`.

This formulation is known as *the Dantzig selector*.


.. rubric:: The conic form

We can rewrite the Dantzig selector problem as:

.. math::

  \begin{aligned}
    & \underset{x}{\text{minimize}} 
    & &  \| x \|_1 \\
    & \text{subject to}
    & &  ( A^* (y - A x ), \delta) \in \LLL^n_{\infty}
  \end{aligned}

where :math:`\LLL^n_{\infty}` is the epigraph of the 
:math:`\ell_{\infty}`-norm. It is a convex cone.

Then the Dantzig selector can be modeled as a standard conic form as follows:

* :math:`f(x) = \| x \|_1`
* :math:`\AAA(x) = (-A^* A x, 0)` is a mapping from :math:`\RR^n \to \RR^{n+1}`
* :math:`b = (A^* y, \delta)`; note that :math:`b \in \RR^{n+1}`
* :math:`\KKK = \LLL^n_{\infty}`

Note carefully that:

.. math::

    \AAA(x) + b = (-A^* A x, 0) + (A^* y, \delta) = (A^* (y - Ax), \delta)

Thus:

.. math::

    \AAA(x) + b \in \KKK = \LLL^n_{\infty} \iff \| A^* (y - Ax) \|_{\infty} \leq \delta


.. rubric:: Dual problem

The dual of the :math:`\LLL^n_{\infty}` cone is the 
:math:`\LLL^n_{1}` cone. 
The dual variable :math:`\lambda` will lie in the dual cone :math:`\LLL^n_{1}`.
It will be easier to work with defining :math:`\lambda = (z, \tau)` such that

.. math::

    \lambda \in \LLL^n_{1}  \iff \| z \|_1 \leq \tau

The convex conjugate of the l1 norm function :math:`f(x) = \| x \|_1`
is the indicator function for the :math:`\ell_{\infty}` norm ball.

.. math:: 

    f^*(x) = I_{\ell_{\infty}}(x) =  \begin{cases} 
    0 & \text{if } \| x \|_{\infty} \leq 1 \\
    \infty       & \text{otherwise}
  \end{cases}

The adjoint of the linear operator is given by:

.. math::

    \AAA^*(\lambda) =  \AAA^*((z, \tau)) = -A^* A z

Plugging into the dual problem definition:

.. math::

    \begin{split}\begin{aligned}
    & \underset{\lambda}{\text{maximize}}
    & & - f^*(\AAA^*(\lambda)) - \langle b, \lambda \rangle \\
    & \text{subject to}
    & &  \lambda \in \KKK^*
    \end{aligned}\end{split}


We get:

.. math::

    \begin{split}\begin{aligned}
    & \underset{z}{\text{maximize}}
    & & - I_{\ell_{\infty}}( -A^* A z) - \langle A^*y, z \rangle - \delta \tau \\
    & \text{subject to}
    & &  (z, \tau) \in \LLL^n_1
    \end{aligned}\end{split}


For :math:`\delta > 0`, the solution :math:`\lambda` will be on the boundary of the
convex cone :math:`\LLL^n_1`, giving us:

.. math::

    \|z \|_1 = \tau

Plugging this back, we get the unconstrained maximization problem:

.. math::

    \underset{z}{\text{maximize}} - I_{\ell_{\infty}}( -A^* A z) - \langle A^*y, z \rangle - \delta \| z \|_1
