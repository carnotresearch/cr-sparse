CR.Sparse
=====================================

A JAX/XLA based library of accelerated models and algorithms for inverse problems in 
sparse representation and compressive sensing. 
`GITHUB <https://github.com/carnotresearch/cr-sparse>`_.

.. panels::
   :card: shadow


   ----
   Wavelets
   ^^^^^^^^^^^^^^
      * Haar, Daubechies, Symlets, Coiflets, Biorthogonal, 
        Reverse biorthogonal, Discrete Meyer
      * dwt, idwt, upcoef, downcoef
      * wavedec, waverec
      * dwt2, idwt2
   ----
   Linear Operators
   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

      * identity, matrix, diagonal, zero, flipud, sum, pad_zeros, 
        symmetrize, restriction
      * running_average, fir_filter
      * fourier_basis, dirac_fourier_basis, cosine_basis, walsh_hadamard_basis
      * gaussian_dict, rademacher_dict, random_onb_dict, random_orthonormal_rows_dict
      * circulant
      * first_derivative, second_derivative
      * neg, scale, add, subtract, compose, transpose, hermitian, hcat, power
   ----
   Dictionaries
   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

     * gaussian, rademacher, dirac fourier, dirac cosine, 

   ----
   Sparse Recovery
   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

      * Greedy Recovery Algorithms: OMP, CoSaMP, SP, IHT, NIHT, HTP, NHTP,
      * Convex Optimization Algorithms: TNIPM, ADMM, 
   ----
   Linear Algebra Routines
   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

      * Triangular systems
   ----
   Evaluation Framework
   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   ----
   Sample Data Generation
   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   ----
   Utilities
   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   intro
   tutorials/index
   start
   fwsp/index
   source/index
   gallery/index
   benchmarks/index
   acronyms
   zzzreference



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
