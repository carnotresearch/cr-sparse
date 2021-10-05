.. _api:cluster:

Data Clustering
==========================


Vector Quantization
------------------------

.. currentmodule:: cr.sparse.cluster.vq

.. autosummary::
  :toctree: _autosummary

  kmeans
  kmeans_jit
  kmeans_with_seed
  kmeans_with_seed_jit
  find_nearest
  find_nearest_jit
  find_assignment
  find_assignment_jit
  find_new_centroids
  find_new_centroids_jit
  
 
Spectral Clustering
------------------------

.. currentmodule:: cr.sparse.cluster.spectral

.. autosummary::
  :toctree: _autosummary

  unnormalized
  unnormalized_k
  unnormalized_k_jit


Data types
--------------------

.. currentmodule:: cr.sparse.cluster

.. autosummary::
  :nosignatures:
  :toctree: _autosummary
  :template: namedtuple.rst

  vq.KMeansState
  vq.KMeansSolution
  spectral.SpectralclusteringSolution

