Functional Models and Algorithms for Sparse Signal Processing


|pypi| |license| |zenodo| |docs| |unit_tests| |coverage| |joss|


Quick Start
=========================

.. contents::
    :depth: 2
    :local:

CR-Sparse is a Python library that enables efficiently solving
a wide variety of sparse representation based signal processing problems.
It is a cohesive collection of sub-libraries working together. Individual
sub-libraries provide functionalities for:
wavelets, linear operators, greedy and convex optimization 
based sparse recovery algorithms, subspace clustering, 
standard signal processing transforms,
and linear algebra subroutines for solving sparse linear systems. 
It has been built using `Google JAX <https://jax.readthedocs.io/en/latest/>`_, 
which enables the same high level
Python code to get efficiently compiled on CPU, GPU and TPU architectures
using `XLA <https://www.tensorflow.org/xla>`_. 

See `here <https://cr-sparse.readthedocs.io/en/latest/intro.html>`_ 
for a more detailed introduction.

See `here <https://cr-sparse.readthedocs.io/en/latest/algorithms.html>`_
for the list of algorithms supported (and planned) in CR-Sparse.


The library includes several packages: 

* Wavelet transforms `cr.sparse.wt <https://cr-sparse.readthedocs.io/en/latest/source/wavelets.html>`_
* Efficient linear operators `cr.sparse.lop <https://cr-sparse.readthedocs.io/en/latest/source/lop.html>`_
* Iterative methods for sparse linear systems `cr.sparse.sls <https://cr-sparse.readthedocs.io/en/latest/source/sls.html>`_
* Redundant dictionaries and sensing matrices `cr.sparse.dict <https://cr-sparse.readthedocs.io/en/latest/source/dict.html>`_
* Solvers for sparse approximation and sparse recovery problems

  * Greedy and shrinkage based methods `cr.sparse.pursuit <https://cr-sparse.readthedocs.io/en/latest/source/pursuit.html>`_
  * Convex optimization based methods `cr.sparse.cvx <https://cr-sparse.readthedocs.io/en/latest/source/cvx_recovery.html>`_

* Sparse subspace clustering `cr.sparse.cluster.ssc <https://cr-sparse.readthedocs.io/en/latest/source/ssc.html>`_

The library also provides

* Some sample data generation utilities `cr.sparse.data <https://cr-sparse.readthedocs.io/en/latest/source/data.html>`_
* Some linear algebra utilities `cr.sparse.la <https://cr-sparse.readthedocs.io/en/latest/source/la.html>`_
* Framework for evaluation of sparse recovery algorithms `cr.sparse.ef <https://cr-sparse.readthedocs.io/en/latest/source/ef.html>`_

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

.. |joss| image:: https://joss.theoj.org/papers/ebd4e5ca27a5db705b1dc382b64e0bed/status.svg
    :alt: JOSS
    :scale: 100%
    :target: https://joss.theoj.org/papers/ebd4e5ca27a5db705b1dc382b64e0bed
