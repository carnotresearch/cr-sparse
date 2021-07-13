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
* Additional utilites



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

Installation
---------------------


Basic installation from PYPI::

    python -m pip install cr-sparse

Installation from GitHub::

    python -m pip install git+https://github.com/carnotresearch/cr-sparse.git


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
