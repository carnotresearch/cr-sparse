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

    \widehat{x} = \text{arg} \min_{x} \frac{1}{2}\| b - A x \|_2^2 + \lambda \| x \|_1 

and its variants (with different regularizations).

Here the objective function is:

.. math::

    f(x) = \frac{1}{2}\| b - A x \|_2^2 + \lambda \| x \|_1

Derivation
------------------

For an identity :math:`A`, the problem reduces to:

.. math::

    \widehat{x} = \text{arg} \min_{x} \frac{1}{2}\| b - x \|_2^2 + \lambda \| x \|_1 

The term on the R.H.S. : 

.. math::

    f(x) = \frac{1}{2}\| b - x \|_2^2 + \lambda \| x \|_1  = \sum_{i=1}^n \left [\frac{1}{2} | b_i - x_i|^2  + \lambda |x_i| \right ]

is separable in the components of :math:`x`.

The scalar function

.. math::

    g(\tau) = \frac{1}{2} | \gamma - \tau|^2  + \lambda |\tau|

has a minimizer given by the soft thresholding function: 

.. math::

    \tau^* = \mathbf{ST}_{\lambda}(\gamma) = \begin{cases} 
     \gamma\left ( 1 -  \frac{\lambda}{|\gamma|} \right ) & \text{for} & |\gamma| \gt \lambda\\
     0 & \text{for} & |\gamma| \le \lambda
    \end{cases}

:cite:`zibulevsky2010l1` show that a similar solution emerges when :math:`A` is unitary too.

:cite:`daubechies2004iterative` introduced a surrogate term:

.. math::

    \text{dist}(x, x_0) = \frac{c}{2} \| x  - x_0 \|_2^2 - \frac{1}{2} \| A x - A x_0 \|_2^2 

For this function to be strictly convex w.r.t. :math:`x`, its Hessian must be positive definite. 
This holds if:

.. math::

    c > \| A^H A \|_2 = \lambda_{\max}(A^H A)

the largest eigen value :math:`A^H A`.

The new surrogate objective function is defined as:

.. math::

    \widetilde{f}(x, x_0) = f(x) + \text{dist}(x, x_0) 
    = \frac{1}{2}\| b - A x \|_2^2 + \lambda \| x \|_1 + \frac{c}{2} \| x  - x_0 \|_2^2 - \frac{1}{2} \| A x - A x_0 \|_2^2

Introducing:

.. math::

    \nu_0 = x_0 + \frac{1}{c} A^H (b - A x_0)

We can rewrite the surrogate objective as:

.. math::

    \widetilde{f}(x, x_0) = \text{Const} + \lambda \| x \|_1 + \frac{c}{2} \| x - \nu_0 \|_2^2

As discussed above, this surrogate objective can be minimized using the soft thresholding operator:

.. math::

    x^* = \mathbf{ST}_{\frac{\lambda}{c} }(\nu_0) = \mathbf{ST}_{\frac{\lambda}{c} }\left (x_0 + \frac{1}{c} A^H (b - A x_0) \right )


Starting with some :math:`x_0`, we can propose an itarative method over a sequence 
:math:`x_i = \{x_0, x_1, x_2, \dots \}` as:

.. math::
    :label: ista_iteration_soft_thresholding

    x_{i+1} = \mathbf{ST}_{\frac{\lambda}{c} }\left (x_i + \frac{1}{c} A^H (b - A x_i)\right )

This is the IST algorithm.

By changing the regularization in :eq:`ista_l1_l2_minimization`, we can derive different IST algorithms with different thresholding 
functions. The version below considers a generalized thresholding function which depends on the regularizer.

.. math::
    :label: ista_iteration_thresholding

    x_{i+1} = \mathbf{T}_{\frac{\lambda}{c} }\left (x_i + \frac{1}{c} A^H (b - A x_i)\right )

Sparsifying Basis
'''''''''''''''''''''''

Often, the signal :math:`x` (e.g. an image) may not be sparse or compressible 
but it has a sparse representation in some basis :math:`B`. We have 

.. math::

    \alpha  = B^H x 

as the representation of :math:`x` in the basis :math:`B`.

The regularization is then applied to the representation :math:`\alpha`.
:eq:`ista_l1_l2_minimization` becomes:

.. math::

    \widehat{x} = \text{arg} \min_{x} \frac{1}{2}\| b - A x \|_2^2 + \lambda \| B^H x \|_1 

We can rewrite this as:

.. math::

    \widehat{\alpha} = \text{arg} \min_{\alpha} \frac{1}{2}\| b - A B \alpha \|_2^2 + \lambda \| \alpha \|_1 

:eq:`ista_iteration_thresholding` changes to:

.. math::

    \alpha_{i+1} = \mathbf{T}_{\frac{\lambda}{c} }\left (\alpha_i + \frac{1}{c} B^H A^H (b - A B \alpha_i)\right )

By substituting :math:`\alpha = B^H x` and :math:`x = B \alpha`, we get:

.. math::

    \alpha_{i+1} = \mathbf{T}_{\frac{\lambda}{c} }\left (B^H x_i + \frac{1}{c} B^H A^H (b - A x_i)\right )

Simplifying further:

.. math::
    :label: ista_iteration_thresholding_basis

    x_{i+1} = B \mathbf{T}_{\frac{\lambda}{c} }\left ( B^H  \left (x_i + \frac{1}{c} A^H (b - A x_i)\right ) \right )

This is the version of IST algorithm with an operator :math:`A` and a basis :math:`B`.

Implementation
----------------------

We introduce the current residual:

.. math::

    r = b - A x

and a step size parameter :math:`\alpha = \frac{1}{c}`.
We also assume that a thresholding function :math:`\mathbf{T}` will be user defined.

This simplifies the iteration to:

.. math::

    x_{i+1} = B \mathbf{T}\left ( B^H \left (x_i + \alpha A^H r_i \right ) \right )


.. rubric:: Algorithm state

Current state of the algorithm is described by following quantities:

.. list-table::
    :header-rows: 1

    * - Term
      - Description
    * - ``x``
      - Current estimate of :math:`x`
    * - ``r``
      - Current residual :math:`r`
    * - ``r_norm_sqr``
      - Squared norm of the residual :math:`\| r \|_2^2`
    * - ``x_change_norm``
      - Change in the norm of :math:`x` given by :math:`\|x_{i+1} - x_{i} \|_2`
    * - ``iterations``
      - Number of iterations of algorithm run so far

.. rubric:: Algorithm initialization

We initialize the algorithm with:

* :math:`x \leftarrow x_0` an initial estimate of solution given by user.
  It can be 0.
* :math:`r \leftarrow b - A x_0`
* Compute ``r_norm_sqr`` 
* Give a very high value to ``x_change_norm`` (since there is no :math:`x_{-1}`).
* ``iterations = 0``


.. rubric:: Algorithm iteration

Following steps are involved in each iteration. These are
directly based on :eq:`ista_iteration_thresholding_basis`.

* Compute gradient:  :math:`g \leftarrow \alpha A^H r`
* Update estimate: :math:`x \leftarrow x + g`
* Transform :math:`\alpha \leftarrow B^H x`
* Threshold: :math:`\alpha \leftarrow \mathbf{T} (\alpha)`
* Inverse transform :math:`x \leftarrow B \alpha`
* Update residual: :math:`r \leftarrow b - A x`
