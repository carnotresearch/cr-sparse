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

Bulk of this library is built using functional programming techniques
which is critical for the generation of efficient numerical codes for CPU
and GPU architectures.

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

A framework for building and composing linear operators has been
provided in ``cr.sparse.lop``. Functionality includes:

* Basic operators: identity, matrix, diagonal, zero, flipud, 
  sum, pad_zeros, symmetrize, restriction, etc.
* Signal processing: fourier_basis_1d, dirac_fourier_basis_1d, etc.
* Random dictionaries: gaussian_dict, rademacher_dict, random_onb_dict, random_orthonormal_rows_dict, etc.
* Operator algebra: neg, scale, add, subtract, compose, transpose, hermitian, hcat, etc.
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
* `PyWavelets <https://github.com/PyWavelets/pywt>`_: The DWT and CWT implementations
  in ``cr.sparse.wt`` are largely derived from it. The filter coefficients for discrete
  wavelets have been ported from C to Python from here.
* `HTP <https://github.com/foucart/HTP>`_ Original implementation of Hard Thresholding
  Pursuit in MATLAB.
* `WaveLab <https://github.com/gregfreeman/wavelab850>`_ This MATLAB package helped a lot in
  initial understanding of DWT implementation.
* `YALL1 <http://yall1.blogs.rice.edu/>`_: This is the original MATLAB implementation of the
  ADMM based sparse recovery algorithm.
* `L1-LS <https://web.stanford.edu/~boyd/l1_ls/>`_ is the original MATLAB implementation of the
  Truncated Newton Interior Points Method for solving the l1-minimization problem.
* `Sparsify <https://www.southampton.ac.uk/engineering/about/staff/tb1m08.page#software>`_ provides
  the MATLAB implementations of IHT, NIHT, AIHT algorithms.
* `Sparse and Redundant Representations: <https://elad.cs.technion.ac.il/wp-content/uploads/2018/02/Matlab-Package-Book-1.zip>`_ 
  From Theory to Applications in Signal and Image Processing book code helped a lot in basic understanding
  of sparse representations.
* `aaren/wavelets <https://github.com/aaren/wavelets>`_ is a decent CWT implementation following
  :cite:`torrence1998practical`. Influenced: ``cr.sparse.wt``.
  

Further Reading
------------------
* `Functional programming <https://en.wikipedia.org/wiki/Functional_programming>`_
* `How to Think in JAX <https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html>`_
* `JAX - The Sharp Bits <https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html>`_


.. bibliography::
   :filter: docname in docnames


`Documentation <https://carnotresearch.github.io/cr-sparse>`_ | 
`Code <https://github.com/carnotresearch/cr-sparse>`_ | 
`Issues <https://github.com/carnotresearch/cr-sparse/issues>`_ | 
`Discussions <https://github.com/carnotresearch/cr-sparse/discussions>`_ |
`Sparse-Plex <https://sparse-plex.readthedocs.io>`_ 
