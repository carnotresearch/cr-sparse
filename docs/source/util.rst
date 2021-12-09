Utilities
==============================

.. contents::
    :depth: 2
    :local:


.. currentmodule:: cr.sparse


Metrics for measuring signal and error levels
---------------------------------------------------------

These functions are available under ``cr.sparse.metrics``.

.. currentmodule:: cr.sparse.metrics



Some checks and utilities for matrices (2D arrays)
----------------------------------------------------------

.. currentmodule:: cr.sparse




Sparse representations
------------------------------------

Following functions analyze or construct representation vectors which are known to be sparse.

.. autosummary::
  :toctree: _autosummary

    nonzero_values
    nonzero_indices
    support
    randomize_rows
    randomize_cols
    largest_indices
    largest_indices_by
    hard_threshold
    hard_threshold_sorted
    hard_threshold_by
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




Basic Signal Information
---------------------------------------------

.. autosummary::
  :toctree: _autosummary

  frequency_spectrum
  power_spectrum
  energy

Basic Signal Processing
-------------------------------

.. autosummary::
  :toctree: _autosummary

  normalize
  interpft


Artificial Noise
-----------------------------------


.. autosummary::
  :toctree: _autosummary

  awgn_at_snr

