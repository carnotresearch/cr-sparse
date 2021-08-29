XLA Accelerated Functional Models and Algorithms for Sparse Representations based Signal Processing
=========================================================================================================

|pypi| |license| |zonodo| |docs| |unttests| |coverage| |codacy|

An `overview <https://carnotresearch.github.io/cr-sparse/intro.html>`_ of the library.

This library aims to provide XLA/JAX based Python implementations for
various algorithms related to:

* Sparse approximation
* Compressive sensing
* Dictionary learning

The library also provides

* Various simple dictionaries and sensing matrices
* Sample data generation utilities
* Framework for evaluation of sparse recovery algorithms

Example usage
----------------

.. rubric:: A greedy pursuit based sparse recovery with synthetic data

Build a Gaussian dictionary/sensing matrix:

.. code:: python

  from jax import random
  import cr.sparse.dict as crdict
  M = 128
  N = 256
  key = random.PRNGKey(0)
  Phi = crdict.gaussian_mtx(key, M,N)

Build a K-sparse signal with Gaussian non-zero entries:

.. code:: python

  import cr.sparse.data as crdata
  import jax.numpy as jnp
  K = 16
  key, subkey = random.split(key)
  x, omega = crdata.sparse_normal_representations(key, N, K, 1)
  x = jnp.squeeze(x)

Build the measurement vector:

.. code:: python

  y = Phi @ x


Import the Compressive Sampling Matching Pursuit sparse recovery solver:

.. code:: python

  from cr.sparse.pursuit import cosamp

Solve the recovery problem:

.. code:: python

  solution =  cosamp.matrix_solve(Phi, y, K)

For the complete set of available solvers, see the documentation.


Citing CR.Sparse
------------------------


To cite this repository:

.. code:: tex

    @software{crsparse2021github,
    author = {Shailesh Kumar},
    title = {{CR.Sparse}: XLA Accelerated Functional Models and Algorithms for Sparse Representations based Signal Processing},
    url = {https://cr-sparse.readthedocs.io/en/latest/},
    version = {0.1.5},
    year = {2021},
    }


`Documentation <https://carnotresearch.github.io/cr-sparse>`_ | 
`Code <https://github.com/carnotresearch/cr-sparse>`_ | 
`Issues <https://github.com/carnotresearch/cr-sparse/issues>`_ | 
`Discussions <https://github.com/carnotresearch/cr-sparse/discussions>`_ |
`Examples <https://github.com/carnotresearch/cr-sparse/blob/master/notebooks/README.rst>`_ |
`Experiments <https://github.com/carnotresearch/cr-sparse/blob/master/notebooks/experiments/README.rst>`_ |
`Sparse-Plex <https://sparse-plex.readthedocs.io>`_


.. |docs| image:: https://readthedocs.org/projects/cr-sparse/badge/?version=latest
    :target: https://cr-sparse.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
    :scale: 100%

.. |unttests| image:: https://github.com/carnotresearch/cr-sparse/actions/workflows/ci.yml/badge.svg
    :alt: Unit Tests
    :scale: 100%
    :target: https://github.com/carnotresearch/cr-sparse/actions/workflows/ci.yml


.. |pypi| image:: https://badge.fury.io/py/cr-sparse.svg
    :alt: PyPI CR-Sparse
    :scale: 100%
    :target: https://badge.fury.io/py/cr-sparse

.. |coverage| image:: https://codecov.io/gh/carnotresearch/cr-sparse/branch/master/graph/badge.svg?token=JZQW6QU3S4
    :alt: Coverage
    :scale: 100%
    :target: https://codecov.io/gh/carnotresearch/cr-sparse


.. |license| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
    :alt: License
    :scale: 100%
    :target: https://opensource.org/licenses/Apache-2.0

.. |codacy| image:: https://app.codacy.com/project/badge/Grade/36905009377e4a968124dabb6cd24aae
    :alt: Codacy Badge
    :scale: 100%
    :target: https://www.codacy.com/gh/carnotresearch/cr-sparse/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=carnotresearch/cr-sparse&amp;utm_campaign=Badge_Grade

.. |zonodo| image:: https://zenodo.org/badge/323566858.svg
    :alt: DOI
    :scale: 100%
   :target: https://zenodo.org/badge/latestdoi/323566858
