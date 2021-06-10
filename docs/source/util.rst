Utilities in cr.sparse module
==============================



.. currentmodule:: cr.sparse

Some checks and utilities for matrices (2D arrays)
----------------------------------------------------------


.. autosummary::
  :toctree: _autosummary

    transpose
    hermitian
    is_matrix
    is_square
    is_symmetric
    is_hermitian
    is_positive_definite
    has_orthogonal_columns
    has_orthogonal_rows
    has_unitary_columns
    has_unitary_rows

Row wise and column wise norms for signal/representation matrices
----------------------------------------------------------------------

.. autosummary::
  :toctree: _autosummary

    norms_l1_cw
    norms_l1_rw
    norms_l2_cw
    norms_l2_rw
    norms_linf_cw
    norms_linf_rw
    sqr_norms_l2_cw
    sqr_norms_l2_rw
    normalize_l1_cw
    normalize_l1_rw
    normalize_l2_cw
    normalize_l2_rw


Sparse representations
------------------------------------

Following functions analyze or construct representation vectors which are known to be sparse.

.. autosummary::
  :toctree: _autosummary

    nonzero_values
    nonzero_indices
    randomize_rows
    randomize_cols
    largest_indices
    hard_threshold
    hard_threshold_sorted
    sparse_approximation
    build_signal_from_indices_and_values
    dynamic_range
    nonzero_dynamic_range


.. rubric:: Sparse representation matrices (row-wise)

.. autosummary::
  :toctree: _autosummary

    largest_indices_rw
    take_along_rows
    sparse_approximation_rw

.. rubric:: Sparse representation matrices (column-wise)

.. autosummary::
  :toctree: _autosummary

    largest_indices_cw
    take_along_cols
    sparse_approximation_cw
