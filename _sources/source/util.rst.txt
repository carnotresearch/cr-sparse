Utilities in cr.sparse module
==============================

.. contents::
    :depth: 2
    :local:


.. currentmodule:: cr.sparse

Array data type utilities
-----------------------------------

.. autosummary::
  :toctree: _autosummary

  promote_arg_dtypes
  check_shapes_are_equal


Utilities for vectors
------------------------------------------

.. autosummary::
  :toctree: _autosummary

  is_scalar
  is_vec
  is_line_vec
  is_row_vec
  is_col_vec
  to_row_vec
  to_col_vec
  vec_unit
  vec_shift_right
  vec_rotate_right
  vec_shift_left
  vec_rotate_left
  vec_shift_right_n
  vec_rotate_right_n
  vec_shift_left_n
  vec_rotate_left_n 
  vec_repeat_at_end
  vec_repeat_at_start
  vec_centered
  vec_unit_jit
  vec_repeat_at_end_jit
  vec_repeat_at_start_jit
  vec_centered_jit










Metrics for measuring signal and error levels
---------------------------------------------------------

These functions are available under ``cr.sparse.metrics``.

.. currentmodule:: cr.sparse.metrics

.. autosummary::
    :toctree: _autosummary

    mean_squared
    mean_squared_error
    root_mean_squared
    root_mse
    normalized_root_mse
    peak_signal_noise_ratio
    signal_noise_ratio


Some checks and utilities for matrices (2D arrays)
----------------------------------------------------------

.. currentmodule:: cr.sparse

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
    off_diagonal_elements
    off_diagonal_min
    off_diagonal_max
    off_diagonal_mean


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


Pairwise Distances
-------------------------

.. autosummary::
  :toctree: _autosummary

  pairwise_sqr_l2_distances_rw
  pairwise_sqr_l2_distances_cw
  pairwise_l2_distances_rw
  pairwise_l2_distances_cw
  pdist_sqr_l2_rw
  pdist_sqr_l2_cw
  pdist_l2_rw
  pdist_l2_cw
  pairwise_l1_distances_rw
  pairwise_l1_distances_cw
  pdist_l1_rw
  pdist_l1_cw
  pairwise_linf_distances_rw
  pairwise_linf_distances_cw
  pdist_linf_rw
  pdist_linf_cw



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


Utilities for ND-Arrays
------------------------------------------

.. autosummary::
  :toctree: _autosummary

    arr_largest_index

