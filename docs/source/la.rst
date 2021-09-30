.. _api:la:

Linear Algebra Subroutines
==================================



Triangular Systems
------------------------


.. currentmodule:: cr.sparse.la

.. autosummary::
  :toctree: _autosummary

    solve_Lx_b
    solve_LTx_b
    solve_Ux_b
    solve_UTx_b
    solve_spd_chol


Fundamental Subspaces
--------------------------

.. autosummary::
  :toctree: _autosummary

    orth
    row_space
    null_space
    left_null_space


Singular Value Decomposition 
---------------------------------------

.. autosummary::
  :toctree: _autosummary

    effective_rank
    effective_rank_from_svd
    singular_values
