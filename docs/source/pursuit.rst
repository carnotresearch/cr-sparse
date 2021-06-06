Sparse Recovery Algorithms
=================================



Greedy pursuit based Algorithms
-------------------------------


.. rubric:: Orthogonal Matching Pursuit

.. currentmodule:: cr.sparse.pursuit.omp

.. autosummary::
  :toctree: _autosummary

    solve
    solve_multi

.. rubric:: Compressive Sampling Matching Pursuit

.. currentmodule:: cr.sparse.pursuit.cosamp

.. autosummary::
  :toctree: _autosummary

    solve


.. rubric:: Subspace Pursuit

.. currentmodule:: cr.sparse.pursuit.sp

.. autosummary::
  :toctree: _autosummary

    solve

.. rubric:: Iterative Hard Thresholding

.. currentmodule:: cr.sparse.pursuit.iht

.. autosummary::
  :toctree: _autosummary

    solve


Data Types
-------------------------------

.. currentmodule:: cr.sparse.pursuit

.. autosummary::
  :nosignatures:
  :toctree: _autosummary
  :template: namedtuple.rst

    RecoverySolution

Utilities
-------------------------------

.. autosummary::
  :toctree: _autosummary

    abs_max_idx
    gram_chol_update
