Quick Start
===================

|pypi| |license| |zenodo| |docs| |unit_tests| |coverage| |joss|


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


.. note::

    If you are on Windows, JAX is not yet officially supported.
    However, you can install an unofficial JAX build for windows
    from https://github.com/cloudhan/jax-windows-builder.
    This works quite well for development purposes.



Examples
----------------

* See the :ref:`examples gallery <gallery>`.
* A more extensive collection of example notebooks is available in the `companion repository <https://github.com/carnotresearch/cr-sparse-companion>`_.
* Some micro-benchmarks are reported `here <https://github.com/carnotresearch/cr-sparse/blob/master/paper/paper.md#runtime-comparisons>`_.


.. note::

    ``cr-sparse`` depends on its sister library `cr-nimble <https://github.com/carnotresearch/cr-nimble>`_.
    Normally, it would be installed automatically as a dependency. 
    You may want to install it directly from GITHUB if you need access to the latest code.

    .. code:: shell

        python -m pip install git+https://github.com/carnotresearch/cr-nimble.git


.. |docs| image:: https://readthedocs.org/projects/cr-sparse/badge/?version=latest
    :target: https://cr-sparse.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |unit_tests| image:: https://github.com/carnotresearch/cr-sparse/actions/workflows/ci.yml/badge.svg
    :alt: Unit Tests
    :target: https://github.com/carnotresearch/cr-sparse/actions/workflows/ci.yml


.. |pypi| image:: https://badge.fury.io/py/cr-sparse.svg
    :alt: PyPI cr-sparse
    :target: https://badge.fury.io/py/cr-sparse

.. |coverage| image:: https://codecov.io/gh/carnotresearch/cr-sparse/branch/master/graph/badge.svg?token=JZQW6QU3S4
    :alt: Coverage
    :target: https://codecov.io/gh/carnotresearch/cr-sparse


.. |license| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
    :alt: License
    :target: https://opensource.org/licenses/Apache-2.0

.. |codacy| image:: https://app.codacy.com/project/badge/Grade/36905009377e4a968124dabb6cd24aae
    :alt: Codacy Badge
    :target: https://www.codacy.com/gh/carnotresearch/cr-sparse/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=carnotresearch/cr-sparse&amp;utm_campaign=Badge_Grade

.. |zenodo| image:: https://zenodo.org/badge/323566858.svg
    :alt: DOI
    :target: https://zenodo.org/badge/latestdoi/323566858

.. |joss| image:: https://joss.theoj.org/papers/ebd4e5ca27a5db705b1dc382b64e0bed/status.svg
    :alt: JOSS
    :target: https://joss.theoj.org/papers/ebd4e5ca27a5db705b1dc382b64e0bed
