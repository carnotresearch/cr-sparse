.. _sls:ista:

Iterative Shrinkage and Thresholding Algorithm
======================================================

.. contents::
    :depth: 2
    :local:

Our description is based on :cite:`daubechies2004iterative, elad2010sparse, zibulevsky2010l1`.

ISTA can be used to solve problems of the form:

.. math::
    :label: ista_l1_l2_minimization

    \widehat{\bx} = \text{arg} \min_{x} \frac{1}{2}\| \bb - \bA \bx \|_2^2 + \lambda \| \bx \|_1 

and its variants (with different regularizations).

Here the objective function is:

.. math::

    f(\bx) = \frac{1}{2}\| \bb - \bA \bx \|_2^2 + \lambda \| \bx \|_1

Derivation
------------------

For an identity :math:`\bA`, the problem reduces to:

.. math::

    \widehat{\bx} = \text{arg} \min_{x} \frac{1}{2}\| \bb - \bx \|_2^2 + \lambda \| \bx \|_1 

The term on the R.H.S. : 

.. math::

    f(\bx) = \frac{1}{2}\| \bb - \bx \|_2^2 + \lambda \| \bx \|_1  
    = \sum_{i=1}^n \left [\frac{1}{2} | b_i - x_i|^2  + \lambda |x_i| \right ]

is separable in the components of :math:`\bx`.

The scalar function

.. math::

    g(\tau) = \frac{1}{2} | \gamma - \tau|^2  + \lambda |\tau|

has a minimizer given by the soft thresholding function: 

.. math::

    \tau^* = \mathbf{ST}_{\lambda}(\gamma) = \begin{cases} 
     \gamma\left ( 1 -  \frac{\lambda}{|\gamma|} \right ) & \text{for} & |\gamma| \gt \lambda\\
     0 & \text{for} & |\gamma| \le \lambda
    \end{cases}

Zibulevsky and Elad in :cite:`zibulevsky2010l1`
show that a similar solution emerges when :math:`\bA` is unitary also.

Daubechies et al. in :cite:`daubechies2004iterative`
introduced a surrogate term:

.. math::

    \text{dist}(\bx, \bx_0) 
    = \frac{c}{2} \| \bx  - \bx_0 \|_2^2 - \frac{1}{2} \| \bA \bx - \bA \bx_0 \|_2^2 

For this function to be strictly convex w.r.t. :math:`\bx`, its Hessian must be positive definite. 
This holds if:

.. math::

    c > \| \bA^H \bA \|_2 = \lambda_{\max}(\bA^H \bA)

the largest eigen value :math:`\bA^H \bA`.

The new surrogate objective function is defined as:

.. math::

    \widetilde{f}(\bx, \bx_0) 
    = f(\bx) + \text{dist}(\bx, \bx_0) 
    = \frac{1}{2}\| \bb - \bA \bx \|_2^2 
    + \lambda \| \bx \|_1 + \frac{c}{2} \| \bx  - \bx_0 \|_2^2 
    - \frac{1}{2} \| \bA \bx - \bA \bx_0 \|_2^2

Introducing:

.. math::

    \nu_0 = \bx_0 + \frac{1}{c} \bA^H (\bb - \bA \bx_0)

we can rewrite the surrogate objective as:

.. math::

    \widetilde{f}(\bx, \bx_0) 
    = \text{Const} + \lambda \| \bx \|_1 + \frac{c}{2} \| \bx - \nu_0 \|_2^2

As discussed above, this surrogate objective can be
minimized using the soft thresholding operator:

.. math::

    \bx^* = \mathbf{ST}_{\frac{\lambda}{c} }(\nu_0) 
    = \mathbf{ST}_{\frac{\lambda}{c} }\left (\bx_0 + \frac{1}{c} \bA^H (\bb - \bA \bx_0) \right )


Starting with some :math:`\bx_0`, we can propose
an iterative method over a sequence 
:math:`\{ \bx_i \} = \{\bx_0, \bx_1, \bx_2, \dots \}` as:

.. math::
    :label: ista_iteration_soft_thresholding

    \bx_{i+1} = \mathbf{ST}_{\frac{\lambda}{c} }\left (\bx_i + \frac{1}{c} \bA^H (\bb - \bA \bx_i)\right )

This is the IST algorithm.

By changing the regularization in :eq:`ista_l1_l2_minimization`,
we can derive different IST algorithms with different thresholding 
functions. 
The version below considers a generalized thresholding function which depends on the regularizer.

.. math::
    :label: ista_iteration_thresholding

    \bx_{i+1} = \mathbf{T}_{\frac{\lambda}{c} }\left (\bx_i + \frac{1}{c} \bA^H (\bb - \bA \bx_i)\right )

Sparsifying Basis
'''''''''''''''''''''''

Often, the signal :math:`\bx` (e.g. an image) may not be sparse or compressible 
but it has a sparse representation in some basis :math:`\bB`.
We have 

.. math::

    \ba  = \bB^H \bx 

as the representation of :math:`\bx` in the basis :math:`\bB`.

The regularization is then applied to the representation :math:`\ba`.
:eq:`ista_l1_l2_minimization` becomes:

.. math::

    \widehat{\bx} = \text{arg} \min_{\bx} \frac{1}{2}\| \bb - \bA \bx \|_2^2
    + \lambda \| \bB^H \bx \|_1 

We can rewrite this as:

.. math::

    \widehat{\ba} 
    = \text{arg} \min_{\ba} \frac{1}{2}\| \bb - \bA \bB \ba \|_2^2 
    + \lambda \| \ba \|_1 

:eq:`ista_iteration_thresholding` changes to:

.. math::

    \ba_{i+1} = \mathbf{T}_{\frac{\lambda}{c} }\left (\ba_i 
    + \frac{1}{c} \bB^H \bA^H (\bb - \bA \bB \ba_i)\right )

By substituting :math:`\ba = \bB^H \bx` and 
:math:`\bx = \bB \ba`, we get:

.. math::

    \ba_{i+1} = \mathbf{T}_{\frac{\lambda}{c} }\left (\bB^H \bx_i 
    + \frac{1}{c} \bB^H \bA^H (\bb - \bA \bx_i)\right )

Simplifying further:

.. math::
    :label: ista_iteration_thresholding_basis

    \bx_{i+1} = \bB \mathbf{T}_{\frac{\lambda}{c} }
    \left ( \bB^H  \left (\bx_i + \frac{1}{c} \bA^H (\bb - \bA \bx_i)\right ) \right )

This is the version of IST algorithm with an operator
:math:`\bA` and a sparsifying basis :math:`\bB`.

Implementation
----------------------

We introduce the current residual:

.. math::

    \br = \bb - \bA \bx

and a step size parameter :math:`\alpha = \frac{1}{c}`.
We also assume that a thresholding function
:math:`\mathbf{T}` will be user defined.

This simplifies the iteration to:

.. math::

    \bx_{i+1} = \bB \mathbf{T}\left ( \bB^H 
    \left (\bx_i + \alpha \bA^H \br_i \right ) \right )


.. rubric:: Algorithm state

Current state of the algorithm is described by following quantities:

.. list-table::
    :header-rows: 1

    * - Term
      - Description
    * - ``x``
      - Current estimate of :math:`\bx`
    * - ``r``
      - Current residual :math:`\br`
    * - ``r_norm_sqr``
      - Squared norm of the residual :math:`\| \br \|_2^2`
    * - ``x_change_norm``
      - Change in the norm of :math:`x` given by :math:`\|\bx_{i+1} - \bx_{i} \|_2`
    * - ``iterations``
      - Number of iterations of algorithm run so far

.. rubric:: Algorithm initialization

We initialize the algorithm with:

* :math:`\bx \leftarrow \bx_0` an initial estimate of solution given by user.
  It can be :math:`\bzero`.
* :math:`\br \leftarrow \bb - \bA \bx_0`
* Compute ``r_norm_sqr`` 
* Give a very high value to ``x_change_norm`` (since there is no :math:`\bx_{-1}`).
* ``iterations = 0``


.. rubric:: Algorithm iteration

Following steps are involved in each iteration. These are
directly based on :eq:`ista_iteration_thresholding_basis`.

* Compute gradient:  :math:`\bg \leftarrow \alpha \bA^H \br`
* Update estimate: :math:`\bx \leftarrow \bx + \bg`
* Transform :math:`\alpha \leftarrow \bB^H \bx`
* Threshold: :math:`\alpha \leftarrow \mathbf{T} (\alpha)`
* Inverse transform :math:`\bx \leftarrow \bB \alpha`
* Update residual: :math:`\br \leftarrow \bb - \bA \bx`
