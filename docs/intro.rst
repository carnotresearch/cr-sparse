Introduction
=====================

This library aims to provide XLA/JAX based Python implementations for
various algorithms related to:

* Sparse approximation
* Compressive sensing
* Dictionary learning

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

Sparse Recovery/Approximation Algorithms
--------------------------------------------

JAX based implementations for the following algorithms are included.

* Orthogonal Matching Pursuit
* Compressive Sampling Matching Pursuit
* Subspace Pursuit
* Iterative Hard Thresholding


.. rubric:: Evaluation Framework

The library also provides

* Various simple dictionaries and sensing matrices
* Sample data generation utilities
* Framework for evaluation of sparse recovery algorithms

.. highlight:: shell


Installation::

    python -m pip install cr-sparse

Installation from GitHub::

    python -m pip install git+https://github.com/carnotresearch/cr-sparse.git


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