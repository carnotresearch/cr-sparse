Functional Models and Algorithms for Sparse Signal Processing   
==================================================================


|pypi| |license| |zenodo| |docs| |unit_tests| |coverage| |joss|


Introduction
-------------------


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

.. image:: docs/images/srr_cs.png

For detailed documentation and usage, please visit `online docs <https://cr-sparse.readthedocs.io/en/latest>`_.

For theoretical background, please check online notes at `Topics in Signal Processing <https://tisp.indigits.com>`_
and references therein (still under development).

``CR-Sparse`` is part of
`CR-Suite <https://carnotresearch.github.io/cr-suite/>`_.

Related libraries:

* `CR-Nimble <https://cr-nimble.readthedocs.io>`_
* `CR-Wavelets <https://cr-wavelets.readthedocs.io>`_


Supported Platforms
----------------------

``CR-Sparse`` can run on any platform supported by ``JAX``. 
We have tested ``CR-Sparse`` on Mac and Linux platforms and Google Colaboratory.

* The latest code in the library has been tested against JAX 0.3.14.
* The last released version of CR-Sparse (0.2.2) was tested against JAX 0.1.55 and later JAX 0.1.x versions. 

``JAX`` is not officially supported on Windows platforms at the moment. 
Although, it is possible to build it from source using Windows Subsystems for Linux.
Alternatively, you can check out the community supported Windows build for JAX
available from https://github.com/cloudhan/jax-windows-builder.
This seems to work well and all the unit tests in the library have passed
on Windows also. 

Installation
-------------------------------

Installation from PyPI:

.. code:: shell

    python -m pip install cr-sparse

Directly from our GITHUB repository:

.. code:: shell

    python -m pip install git+https://github.com/carnotresearch/cr-sparse.git



Examples/Usage
----------------

See the `examples gallery <https://cr-sparse.readthedocs.io/en/latest/gallery/index.html>`_ in the documentation.
Here is a small selection of examples:

* `Sparse recovery using Truncated Newton Interior Points Method <https://cr-sparse.readthedocs.io/en/latest/gallery/rec_l1/spikes_l1ls.html>`_ 
* `Sparse recovery with ADMM <https://cr-sparse.readthedocs.io/en/latest/gallery/rec_l1/partial_wh_sensor_cosine_basis.html>`_ 
* `Compressive sensing operators <https://cr-sparse.readthedocs.io/en/latest/gallery/lop/cs_operators.html>`_ 
* `Image deblurring with LSQR and FISTA algorithms <https://cr-sparse.readthedocs.io/en/latest/gallery/lop/deblurring.html>`_ 
* `Deconvolution of the effects of a Ricker wavelet <https://cr-sparse.readthedocs.io/en/latest/gallery/lop/deconvolution.html>`_ 
* `Wavelet transform operators <https://cr-sparse.readthedocs.io/en/latest/gallery/lop/wt_op.html>`_ 
* `CoSaMP step by step <https://cr-sparse.readthedocs.io/en/latest/gallery/pursuit/cosamp_step_by_step.html>`_ 


A more extensive collection of example notebooks is available in the `companion repository <https://github.com/carnotresearch/cr-sparse-companion>`_.
Some micro-benchmarks are reported `here <https://github.com/carnotresearch/cr-sparse/blob/master/paper/paper.md#runtime-comparisons>`_.


Contribution Guidelines/Code of Conduct
----------------------------------------

* `Contribution Guidelines <CONTRIBUTING.md>`_
* `Code of Conduct <CODE_OF_CONDUCT.md>`_

Citing CR-Sparse
------------------------


To cite this library:

.. code:: tex

    @article{Kumar2021,
      doi = {10.21105/joss.03917},
      url = {https://doi.org/10.21105/joss.03917},
      year = {2021},
      publisher = {The Open Journal},
      volume = {6},
      number = {68},
      pages = {3917},
      author = {Shailesh Kumar},
      title = {CR-Sparse: Hardware accelerated functional algorithms for sparse signal processing in Python using JAX},
      journal = {Journal of Open Source Software}
    }




`Documentation <https://carnotresearch.github.io/cr-sparse>`_ | 
`Code <https://github.com/carnotresearch/cr-sparse>`_ | 
`Issues <https://github.com/carnotresearch/cr-sparse/issues>`_ | 
`Discussions <https://github.com/carnotresearch/cr-sparse/discussions>`_ |


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
