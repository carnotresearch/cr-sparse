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


Subspaces
----------------

.. currentmodule:: cr.sparse.la.subspaces

.. autosummary::
  :toctree: _autosummary

  principal_angles_cos
  principal_angles_rad
  principal_angles_deg
  smallest_principal_angle_cos
  smallest_principal_angle_rad
  smallest_principal_angle_deg
  smallest_principal_angles_cos

