Introduction
=====================

.. contents::
    :depth: 2
    :local:


This library aims to provide XLA/JAX based Python implementations for
various algorithms related to:

* Sparse approximation :cite:`mallat2008wavelet,elad2010sparse`
* Compressive sensing :cite:`donoho2006compressed,candes2006compressive,candes2008introduction,baraniuk2011introduction`
* Linear operators

.. image:: images/srr_cs.png

Bulk of this library is built using functional programming techniques
which is critical for the generation of efficient numerical codes for CPU
and GPU architectures.

Sparse approximation and recovery problems
------------------------------------------------

In the sparse approximation problems :cite:`mallat2008wavelet,elad2010sparse`, we have a 
dictionary of atoms designed for a class of signals
such that the dictionary enables us to construct
a sparse representation of the signal. The
sparse and redundant representation model is:

.. math::

    x = \mathcal{D} \alpha + \eta

where :math:`x \in \mathbb{R}^M` is a single from the given 
class of signals, :math:`\mathcal{D} \in \mathbb{R}^{M \times N}`
is a dictionary consisting of :math:`N` atoms (column vectors) chosen
specifically for the class of signals, :math:`\alpha`
is the sparse representation of :math:`x` in :math:`\mathcal{D}`
giving us an approximation :math:`\hat{x} = \mathcal{D} \alpha`
and :math:`\eta` is the approximation error. The
dictionary :math:`\mathcal{D}` is called the sparsifying dictionary.
The sparse approximation problem consists of finding
the best sparse :math:`\alpha` for a given :math:`x`.

In the compressed sensing (CS) setting,
a sparse signal :math:`x \in \mathbb{R}^N` is captured 
through :math:`M \ll N` linear measurements which are
sufficient to recover :math:`x` from the measurements.
The model is given by:

.. math::

    y = \Phi x + e

where :math:`y \in \mathbb{R}^M` is the vector of :math:`M` linear
measurements on :math:`x`, :math:`\Phi \in \mathbb{R}^{M \times N}` 
is the sensing matrix [or measurement matrix] whose
rows represent the linear functionals on :math:`x`, :math:`x \in \mathbb{R}^N`
is the sparse signal being measured and :math:`e` is the measurement
noise. Typically, :math:`x` by itself is not sparse but it has
a sparse representation in a sparsifying basis :math:`\Psi`
as :math:`x = \Psi \alpha`. The model then becomes:

.. math::

    y = \Phi \Psi \alpha + e.

Sparse recovery consists of finding :math:`\alpha` from
:math:`y` with minimum number of measurements possible.

Both sparse recovery and sparse approximation problems
can be addressed by same algorithms (though their 
performance analysis is different). To simplify the
notation, we will refer to :math:`\mathcal{D}` or :math:`\Phi` 
or :math:`\Phi \Psi` collectively as :math:`A` and attempt to
solve the under-determined system :math:`y = A x + e`
with the prior on the solution that very few entries
in :math:`x` are non-zero. In general, we assume that
:math:`A` is full rank, unless otherwise specified.

The indices of non-zero
entry of :math:`x` form the support of :math:`x`. Corresponding
columns in :math:`A` participate in the sparse
representation of :math:`y`. We can call these columns
also as the support of :math:`x`. 

.. math::

  \mathop{\mathrm{supp}}(x) \triangleq \{i : x_i \neq 0 \}.

Recovering the representation :math:`x`
involves identifying its support :math:`\Lambda = \mathop{\mathrm{supp}}(x)`
and identifying the non-zero entries over the support.
If the support has been 
correctly identified, a straight-forward
way to get the non-zero entries is to compute the
least squares solution :math:`A_{\Lambda}^{\dag} y`.
The :math:`\ell_0` norm of :math:`x` denoted by :math:`\| x\|_0` 
is the number of non-zero entries in :math:`x`.
A representation :math:`y = A x`
is sparse if :math:`\| x\|_0 \ll N`.
An algorithm which can
obtain such  a representation is called a *sparse coding
algorithm*.


:math:`\ell_0` problems
'''''''''''''''''''''''''''''''''

The :math:`K`-SPARSE approximation can be formally expressed as:

.. math::

  \begin{aligned}
    & \underset{x}{\text{minimize}} 
    & &  \| y - A x \|_2 \\
    & \text{subject to}
    & &  \| x \|_0 \leq K.
  \end{aligned}

If the measurements are noiseless, we are interested in 
exact recovery. 
The :math:`K`-EXACT-SPARSE approximation can be formally expressed as:

.. math::

  \begin{aligned}
    & \underset{x}{\text{minimize}} 
    & &  \| x \|_0 \\
    & \text{subject to}
    & &  y = \Phi x\\
    & \text{and}
    & &  \| x \|_0 \leq K
  \end{aligned}


We need to discover both the sparse support for :math:`x` and
the non-zero values over this support. A greedy algorithm
attempts to guess the support incrementally and solves
a smaller (typically least squares) subproblem to estimate
the nonzero values on this support. It then computes the
residual :math:`r = y - A x` and analyzes the correlation of :math:`r`
with the atoms in :math:`A`, via the vector :math:`h = A^T r`, to
improve its guess for the support and update :math:`x` accordingly.


:math:`\ell_1` problems
''''''''''''''''''''''''''''''''


We introduce the different :math:`\ell_1` minimization problems supported by the
``cr.sparse.cvx.admm`` package.

The :math:`\ell_0` problems are not convex. Obtaining a global minimizer 
is not feasible (NP hard). One way around is to use convex relaxation
where a cost function is replaced by its convex version. 
For :math:`\| x \|_0`, the closest convex function is :math:`\| x \|_1` 
or :math:`\ell_1` norm. With this, the exact-sparse recovery problem becomes

.. math::

  {\min}_{x} \| x\|_{1} \; \text{s.t.} \, A x = b


This problem is known as Basis Pursuit (BP) in literature. It can be shown 
that under appropriate conditions on :math:`A`, the basis pursuit solution
coincides with the exact sparse solution. In general, :math:`\ell_1`-norm
minimization problems tend to give sparse solutions.

If :math:`x` is sparse in an sparsifying basis :math:`\Psi` as :math:`x  = \Psi \alpha`
(i.e. :math:`\alpha` is sparse rather than :math:`x`), then we can adapt the
BP formulation as

.. math::

  {\min}_{x} \| W x\|_{1} \; \text{s.t.} \, A x = b

where :math:`W = \Psi^T` and :math:`A` is the sensing matrix :math:`\Phi`.

Finally, in specific problems, different atoms of :math:`\Psi` may 
have different importance. In this case, the :math:`\ell_1` norm
may be adapted to reflect this importance by a non-negative weight vector :math:`w`:

.. math::

  \| \alpha \|_{w,1} = \sum_{i=1}^{N} w_i | \alpha_i |.

This is known as the weighted :math:`\ell_1` semi-norm.

This gives us the general form of the basis pursuit problem

.. math::

  \tag{BP}
  {\min}_{x} \| W x\|_{w,1} \; \text{s.t.} \, A x = b


Usually, the measurement process introduces noise. Thus, 
a constraint :math:`A x = b` is too strict. We can relax this 
to allow for presence of noise as :math:`\| A x - b \|_2 \leq \delta`
where :math:`\delta` is an upper bound on the norm of the measurement noise
or approximation error. 
This gives us the Basis Pursuit with Inequality Constraints (BPIC) problem:

.. math::

  {\min}_{x} \| x\|_{1} \; \text{s.t.} \, \| A x - b \|_2 \leq \delta

The more general form is the L1 minimization problem with L2 constraints:

.. math::

  \tag{L1/L2con}
  {\min}_{x} \| W x\|_{w,1} \; \text{s.t.} \, \| A x - b \|_2 \leq \delta

The constrained BPIC problem can be transformed into an equivalent 
unconstrained convex problem:

.. math::

  {\min}_{x} \| x\|_{1} + \frac{1}{2\rho}\| A x - b \|_2^2.

This is known as Basis Pursuit Denoising (BPDN) in literature.
The more general form is the L1/L2 minimization:

.. math::

  \tag{L1/L2}
  {\min}_{x} \| W x\|_{w,1} + \frac{1}{2\rho}\| A x - b \|_2^2 

We also support corresponding non-negative counter-parts.
The nonnegative basis pursuit problem:

.. math::
  \tag{BP+}
  {\min}_{x} \| W x\|_{w,1} \; \text{s.t.} \, A x = b \, \, \text{and} \, x \succeq 0

The nonnegative L1/L2 minimization or basis pursuit denoising problem:

.. math::

  \tag{L1/L2+}
  {\min}_{x} \| W x\|_{w,1} + \frac{1}{2\rho}\| A x - b \|_2^2  \; \text{s.t.} \, x \succeq 0

The nonnegative L1 minimization problem with L2 constraints:

.. math::

  \tag{L1/L2con+}
  {\min}_{x} \| W x\|_{w,1} \; \text{s.t.} \, \| A x - b \|_2 \leq \delta \, \, \text{and} \, x \succeq 0


Functional Programming
---------------------------


Functional Programming is a programming paradigm where computer programs are constructed 
by applying and composing functions. Functions define a tree of expressions which 
map values to other values (akin to mathematical functions) rather than a sequence
of iterative statements. Some famous languages based on functional programming are
Haskell and Common Lisp.
A key idea in functional programming is a *pure function*. 
A pure function has following properties: 

* The return values are identical for identical arguments.
* The function has no side-effects (no mutation of local static variables, 
  non-local variables, etc.). 


XLA is a domain-specific compiler for linear algebra. 
XLA uses JIT (just-in-time) compilation techniques to analyze the structure of a 
numerical algorithm written using it.
It then specializes the algorithm for actual runtime dimensions and types of parameters involved,
fuses multiple operations together and emits efficient native machine code for
devices like CPUs, GPUs and custom accelerators (like Google TPUs).

JAX is a front-end for XLA and Autograd
with a NumPy inspired API.
Unlike NumPy, JAX arrays are always immutable. While ``x[0] = 10`` is perfectly fine
in NumPy as arrays are mutable, the equivalent functional code in JAX is
``x = x.at[0].set(10)``.


Linear Operators
-----------------------------------------

Efficient linear operator implementations provide much faster
computations compared to direct matrix vector multiplication.
PyLops :cite:`ravasi2019pylops` is a popular collection of
linear operators implemented in Python. 

A linear operator :math:`T : X \to Y` connects a model space :math:`X` 
to a data space :math:`Y`.

A linear operator satisfies following laws:

.. math::

    T (x + y) = T (x) + T (y)

and

.. math::

    T (\alpha x) = \alpha T(x)

Thus, for a general linear combination:

.. math::

    T (\alpha x + \beta y) = \alpha T (x) + \beta T (y)

We are concerned with linear operators :math:`T : \mathbb{F}^n \to \mathbb{F}^m`
where :math:`\mathbb{F}` is either the field of real numbers or 
complex numbers. 
:math:`X = \mathbb{F}^n` is the model space and 
:math:`Y = \mathbb{F}^m` is the data space.
Such a linear operator can be represented by a two dimensional matrix :math:`A`.
The forward operation is given by:

.. math::

    y = A x.

The corresponding adjoint operation is given by:

.. math::

    \hat{x} = A^H y

We represent a linear operator by a pair of functions ``times`` and ``trans``. 
The ``times`` function implements the forward operation while the ``trans``
function implements the adjoint operation.

An inverse problem consists of computing :math:`x` given :math:`y` and :math:`A`.

A framework for building and composing linear operators has been
provided in ``cr.sparse.lop``. Functionality includes:

* Basic operators: identity, matrix, diagonal, zero, flipud, 
  sum, pad_zeros, symmetrize, restriction, etc.
* Signal processing: fourier_basis_1d, dirac_fourier_basis_1d, etc.
* Random dictionaries: gaussian_dict, rademacher_dict, random_onb_dict, random_orthonormal_rows_dict, etc.
* Operator calculus: neg, scale, add, subtract, compose, transpose, hermitian, hcat, etc.
* Additional utilities



Greedy Sparse Recovery/Approximation Algorithms
------------------------------------------------

JAX based implementations for the following algorithms are included.

* Orthogonal Matching Pursuit :cite:`pati1993orthogonal,tropp2004greed`
* Compressive Sampling Matching Pursuit :cite:`needell2009cosamp`
* Subspace Pursuit :cite:`dai2009subspace`
* Iterative Hard Thresholding :cite:`blumensath2009iterative`
* Hard Thresholding Pursuit :cite:`foucart2011recovering`

Convex Optimization based Recovery Algorithms
-----------------------------------------------------

Convex optimization :cite:`boyd2004convex` based methods provide more 
reliable solutions to sparse recovery problems although they tend to be
computationally more complex. 
The first method appeared around 1998 as basis pursuit :cite:`chen1998atomic`.

Alternating directions :cite:`boyd2011distributed` based methods provide
simple yet efficient iterative solutions for sparse recovery. 

:cite:`yang2011alternating` describes inexact ADMM based solutions 
for a variety of :math:`\ell_1` minimization problems. The authors
provide a MATLAB package ``yall1`` :cite:`zhang2010user`. 
A port of ``yall1`` (Your algorithms for :math:`\ell_1`) has been provided.
It provides alternating directions method of multipliers based solutions for
basis pursuit, basis pursuit denoising, basis pursuit with inequality constraints,
their non-negative counterparts and other variants.



Evaluation Framework
--------------------------

The library also provides

* Various simple dictionaries and sensing matrices
* Sample data generation utilities
* Framework for evaluation of sparse recovery algorithms

.. highlight:: shell


Open Source Credits
-----------------------------

Major parts of this library are directly influenced by existing projects.
While the implementation in CR-Sparse is fresh (based on JAX), it has been
possible thanks to the extensive study of existing implementations. We list
here some of the major existing projects which have influenced the implementation
in CR-Sparse. Let us know if we missed anything. 

* `JAX <https://github.com/google/jax>`_ The overall project structure is heavily
  influenced by the conventions followed in JAX. We learned the functional programming
  techniques as applicable for linear algebra work by reading the source code of JAX.
* `SciPy <https://github.com/scipy/scipy>`_ JAX doesn't have all parts of SciPy ported
  yet. Some parts of SciPy have been adapted and re-written (in functional manner) 
  as per the needs of CR-Sparse. E.g. ``cr.sparse.dsp.signals``. The :cite:`torrence1998practical` version
  of CWT in ``cr.sparse.wt``.
* `OpTax <https://github.com/deepmind/optax>`_  This helped in understanding how to 
  use Named Tuples as states for iterative algorithms.  This was also useful 
  in conceptualizing the structure for ``cr.sparse.lop``. 
* `PyLops <https://github.com/PyLops/pylops>`_: The ``cr.sparse.lop`` library is 
  heavily influenced by it.
* `HTP <https://github.com/foucart/HTP>`_ Original implementation of Hard Thresholding
  Pursuit in MATLAB.
* `YALL1 <http://yall1.blogs.rice.edu/>`_: This is the original MATLAB implementation of the
  ADMM based sparse recovery algorithm.
* `L1-LS <https://web.stanford.edu/~boyd/l1_ls/>`_ is the original MATLAB implementation of the
  Truncated Newton Interior Points Method for solving the l1-minimization problem.
* `Sparsify <https://www.southampton.ac.uk/engineering/about/staff/tb1m08.page#software>`_ provides
  the MATLAB implementations of IHT, NIHT, AIHT algorithms.
* `Sparse and Redundant Representations: <https://elad.cs.technion.ac.il/wp-content/uploads/2018/02/Matlab-Package-Book-1.zip>`_ 
  From Theory to Applications in Signal and Image Processing book code helped a lot in basic understanding
  of sparse representations.
  

Further Reading
------------------
* `Functional programming <https://en.wikipedia.org/wiki/Functional_programming>`_
* `How to Think in JAX <https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html>`_
* `JAX - The Sharp Bits <https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html>`_


`Documentation <https://carnotresearch.github.io/cr-sparse>`_ | 
`Code <https://github.com/carnotresearch/cr-sparse>`_ | 
`Issues <https://github.com/carnotresearch/cr-sparse/issues>`_ | 
`Discussions <https://github.com/carnotresearch/cr-sparse/discussions>`_ |
`Sparse-Plex <https://sparse-plex.readthedocs.io>`_ 
