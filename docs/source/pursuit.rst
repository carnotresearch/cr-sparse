Sparse Recovery Algorithms
=================================



Greedy pursuit based Algorithms
-------------------------------

.. currentmodule:: cr.sparse.pursuit


.. rubric:: Matching Pursuit Based Algorithms

.. autosummary::
  :toctree: _autosummary

    omp.solve
    omp.solve_multi

.. rubric:: Compressive Sensing Matching Pursuit (CSMP) Algorithms

.. autosummary::
  :toctree: _autosummary

    cosamp.solve
    sp.solve

.. rubric:: Hard Thresholding Based Algorithms

.. autosummary::
  :toctree: _autosummary

    iht.solve
    htp.solve


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
