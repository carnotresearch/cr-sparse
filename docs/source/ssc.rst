Sparse Subspace Clustering
=================================================


.. currentmodule:: cr.sparse.cluster.ssc

SSC-OMP
--------------

.. autosummary::
  :toctree: _autosummary

  build_representation_omp
  build_representation_omp_jit
  batch_build_representation_omp
  batch_build_representation_omp_jit


.. rubric:: Utility functions

.. autosummary::
  :toctree: _autosummary

  sparse_to_full_rep
  sparse_to_bcoo
  bcoo_to_sparse
  bcoo_to_sparse_jit
  rep_to_affinity

Metrics for quality of sparse subspace clustering
------------------------------------------------------


.. autosummary::
  :toctree: _autosummary

  subspace_preservation_stats
  subspace_preservation_stats_jit
  sparse_subspace_preservation_stats
  sparse_subspace_preservation_stats_jit


Tools for analyzing data (with ground truth) 
------------------------------------------------------

.. autosummary::
  :toctree: _autosummary


  angles_between_points
  min_angles_inside_cluster
  min_angles_outside_cluster
  nearest_neighbors_inside_cluster
  nearest_neighbors_outside_cluster
  sorted_neighbors
  inn_positions


Examples
-----------------

* :ref:`gallery:cluster:ssc:omp`



