Functional Models and Algorithms for Sparse Signal Processing


|pypi| |license| |zenodo| |docs| |unit_tests| |coverage|


Quick Start
=========================

An `overview <https://cr-sparse.readthedocs.io/en/latest/intro.html>`_ of the library.

.. contents::
    :depth: 2
    :local:


This library aims to provide XLA/JAX based Python implementations for
various models and algorithms related to:

* Wavelet transforms
* Efficient linear operators
* Iterative methods for sparse linear systems
* Redundant dictionaries
* Sparse approximations on redundant dictionaries

  * Greedy methods
  * Convex optimization based methods
  * Shrinkage methods

* Sparse recovery from compressive sensing based measurements

  * Greedy methods
  * Convex optimization based methods


The library also provides

* Various simple dictionaries and sensing matrices
* Sample data generation utilities
* Framework for evaluation of sparse recovery algorithms

Examples
----------------

Some micro-benchmarks are reported `here <https://github.com/carnotresearch/cr-sparse/blob/master/paper/paper.md#runtime-comparisons>`_.
Jupyter notebooks for these benchmarks are in the `companion repository <https://github.com/carnotresearch/cr-sparse-companion>`_.


See the `examples gallery <https://cr-sparse.readthedocs.io/en/latest/gallery/index.html>`_ for an 
extensive set of examples. Here is a small selection of examples:

* `Sparse recovery using Truncated Newton Interior Points Method <https://cr-sparse.readthedocs.io/en/latest/gallery/rec_l1/spikes_l1ls.html>`_ 
* `Sparse recovery with ADMM <https://cr-sparse.readthedocs.io/en/latest/gallery/rec_l1/partial_wh_sensor_cosine_basis.html>`_ 
* `Compressive sensing operators <https://cr-sparse.readthedocs.io/en/latest/gallery/lop/cs_operators.html>`_ 
* `Image deblurring with LSQR and FISTA algorithms <https://cr-sparse.readthedocs.io/en/latest/gallery/lop/deblurring.html>`_ 
* `Deconvolution of the effects of a Ricker wavelet <https://cr-sparse.readthedocs.io/en/latest/gallery/lop/deconvolution.html>`_ 
* `Wavelet transform operators <https://cr-sparse.readthedocs.io/en/latest/gallery/lop/wt_op.html>`_ 
* `CoSaMP step by step <https://cr-sparse.readthedocs.io/en/latest/gallery/pursuit/cosamp_step_by_step.html>`_ 


A more extensive collection of example notebooks is available in the `companion repository <https://github.com/carnotresearch/cr-sparse-companion>`_.


Platform Support
----------------------

``cr-sparse`` can run on any platform supported by ``JAX``. 
We have tested ``cr-sparse`` on Mac and Linux platforms and Google Colaboratory.

``JAX`` is not officially supported on Windows platforms at the moment. 
Although, it is possible to build it from source using Windows Subsystems for Linux.

Installation
-------------------------------

Installation from PyPI:

.. code:: shell

    python -m pip install cr-sparse

Directly from our GITHUB repository:

.. code:: shell

    python -m pip install git+https://github.com/carnotresearch/cr-sparse.git


Citing cr-sparse
------------------------


To cite this repository:

.. code:: tex

    @software{crsparse2021github,
    author = {Shailesh Kumar},
    title = {{cr-sparse}: Functional Models and Algorithms for Sparse Signal Processing},
    url = {https://cr-sparse.readthedocs.io/en/latest/},
    version = {0.1.6},
    year = {2021},
    doi={10.5281/zenodo.5322044},
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

.. |unit_tests| image:: https://github.com/carnotresearch/cr-sparse/actions/workflows/ci.yml/badge.svg
    :alt: Unit Tests
    :scale: 100%
    :target: https://github.com/carnotresearch/cr-sparse/actions/workflows/ci.yml


.. |pypi| image:: https://badge.fury.io/py/cr-sparse.svg
    :alt: PyPI cr-sparse
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

.. |zenodo| image:: https://zenodo.org/badge/323566858.svg
    :alt: DOI
    :scale: 100%
    :target: https://zenodo.org/badge/latestdoi/323566858
