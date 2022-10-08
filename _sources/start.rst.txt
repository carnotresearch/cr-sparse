Quick Start
===================

|pypi| |license| |zenodo| |docs| |unit_tests| |coverage| |joss|


.. contents::
    :depth: 2
    :local:


Installation
-------------------------------

Installation from PyPI:

.. code:: shell

    python -m pip install cr-sparse



Directly from our GITHUB repository:

.. code:: shell

    python -m pip install git+https://github.com/carnotresearch/cr-sparse.git


Platform Support
'''''''''''''''''''''''

``cr-sparse`` can run on any platform supported by ``JAX``. 
We have tested ``cr-sparse`` on Mac and Linux platforms and Google Colaboratory.

``JAX`` is not officially supported on Windows platforms at the moment. 
Although, it is possible to build it from source using Windows Subsystems for Linux.

.. note::

    If you are on Windows, JAX is not yet officially supported.
    However, you can install an unofficial JAX build for windows
    from https://github.com/cloudhan/jax-windows-builder.
    This works quite well for development purposes.


Dependencies
'''''''''''''''''''''''

``cr-sparse`` depends on its sister libraries 

#. `cr-nimble <https://github.com/carnotresearch/cr-nimble>`_
#. `cr-wavelets <https://github.com/carnotresearch/cr-wavelets>`_

Normally, they would be installed automatically as a dependency. 
You may want to install them directly from GITHUB
if you need access to the latest code.

.. code:: shell

    python -m pip install git+https://github.com/carnotresearch/cr-nimble.git
    python -m pip install git+https://github.com/carnotresearch/cr-wavelets.git



Usage
---------------


Examples
'''''''''''''''''''''''

* See the :ref:`examples gallery <gallery>`.
* A more extensive collection of example notebooks is available in the `companion repository <https://github.com/carnotresearch/cr-sparse-companion>`_.
* Some micro-benchmarks are reported `here <https://github.com/carnotresearch/cr-sparse/blob/master/paper/paper.md#runtime-comparisons>`_.



Common Workflows
'''''''''''''''''''''''

.. rubric:: Compressive Sensing

#. Select/Load/Generate a signal to be compressive sampled. 
   
   * :ref:`api:data`
   * :ref:`api:problems`

#. Select a sensing matrix/operator. 
   
   * :ref:`api:dict`
   * :ref:`api:lop`
   * :ref:`api:problems`

#. Compute the measurements.
#. Select a sparsifying basis under which the signal has a sparse representation.

   * :ref:`api:dict`
   * :ref:`api:lop`
   * :ref:`api:problems`
   * `cr-wavelets <https://github.com/carnotresearch/cr-wavelets>`_

#. Combine the sensing matrix and sparsifying basis to generate a linear
   operator which will be used for solving the sparse reconstruction problem.

   * ``compose`` in :ref:`api:lop`

#. Use a sparse recovery algorithm to solve the sparse recovery problem.

   * :ref:`sec:algorithms`
   * :ref:`api:pursuit`
   * :ref:`api:l1min`

#. Use the sparse representation so constructed to generate the approximation
   of the original signal using the sparsifying basis.
#. Measure the quality of reconstruction.
   
   * ``cr.nimble.metrics``


Common Tasks
'''''''''''''''''''''''

Make sure to configure JAX for 64-bit numbers::

    from jax.config import config
    config.update("jax_enable_x64", True)


Essential library imports::

    # jax numpy
    import jax.numpy as jnp
    # cr-nimble library
    import cr.nimble as crn
    # cr-sparse library
    import cr.sparse as crs

You will often need PRNG (Pseudorandom) keys::

    from jax import random
    key = random.PRNGKey(0) # you can put any integer as seed


.. rubric:: Sample data

A sparse signal with normal distributed nonzero values::

    import cr.sparse.data as crdata
    x, omega = crdata.sparse_normal_representations(key, n, k)

A sparse signal consisting of signed spikes::

    x, omega = crdata.sparse_spikes(key, n, k)


Block sparse signal with intra block correlation::

    x, blocks, indices  = crdata.sparse_normal_blocks(
        key, n, k, blk_size, cor=0.9)


Some standard signals based on Wavelab::

    import cr.nimble.dsp.signals as signals
    t, x = signals.heavi_sine()
    t, x = signals.bumps()
    t, x = signals.blocks()
    t, x = signals.ramp()
    t, x = signals.cusp()
    t, x = signals.sing()
    t, x = signals.hi_sine()
    t, x = signals.lo_sine()
    t, x = signals.lin_chirp()
    t, x = signals.two_chirp()
    t, x = signals.quad_chirp()
    t, x = signals.mish_mash()
    t, x = signals.werner_sorrows()


.. rubric:: Sensing Matrices

A Gaussian sensing matrix::

    import cr.sparse.dict as crdict
    Phi = crdict.gaussian_mtx(key, m, n)


A Gaussian sensing matrix operator::

    import cr.sparse.lop as crlop
    Phi = crlop.gaussian_dict(key, m, n)


Computing measurements::

    b = Phi @ x # for matrices
    b = Phi.times(x) # for operators


Haar wavelet basis::

    import cr.sparse.lop as crlop
    Psi = crlop.dwt(n, wavelet='haar', level=5, basis=True)


Computing the representation of a signal in an orthonormal basis::

    alpha = Psi.trans(x)

Constructing a signal from its representation::

    x = Psi.times(alpha)

Creating a composite linear operator of a sensing matrix and a basis::

    A = crlop.compose(Phi, Psi)


Finding how many large coefficients are sufficient to capture
most of the energy in a signal::

    crn.num_largest_coeffs_for_energy_percent(x, 99)


.. rubric:: More Dictionaries

Fourier Heaviside dictionary::

    import cr.sparse.lop as crlop
    heaviside = crlop.heaviside(n)
    fourier_basis = crlop.fourier_basis(n)
    dictionary = crlop.hcat(fourier_basis, heaviside)


Dirac Cosine dictionary::

    dirac_basis = crlop.identity(n)
    cosine_basis = crlop.cosine_basis(n)
    dirac_cosine_basis = crlop.hcat(dirac_basis, cosine_basis)

Dirac Fourier dictionary::

    dirac_basis = crlop.identity(n)
    fourier_basis = crlop.fourier_basis(n)
    dirac_fourier_basis = crlop.hcat(dirac_basis, fourier_basis)

Daubechies basis::

    db_basis = crlop.dwt(n, wavelet='db8', level=level, basis=True)


.. rubric:: Sparse Recovery


Solving a sparse representation problem :math:`\bb = \bA \bx`
using Subspace Pursuit::

    import cr.sparse.pursuit.sp as sp
    sol = sp.solve(A, b, k)
    x = sol.x

Solving using Compressive Sampling Matching Pursuit::

    import cr.sparse.pursuit.cosamp as cosamp
    sol = cosamp.solve(A, b, k)

Solving using Iterative Hard Thresholding::

    from cr.sparse.pursuit import iht
    sol = iht.solve(A, b, k)

Solving using Hard Thresholding Pursuit::

    from cr.sparse.pursuit import iht
    sol = htp.solve(A, b, k)


Solving using SPGL1 algorithm::

    import cr.sparse.cvx.spgl1 as crspgl1
    options = crspgl1.SPGL1Options()
    sol = crspgl1.solve_bp(A, b, options=options)
    sol = crspgl1.solve_bpic(A, b, sigma, options=options)
    sol = crspgl1.solve_lasso(A, b, tau, options=options)


Solving using L1LS algorithm::

    import cr.sparse.cvx.l1ls as l1ls
    sol = l1ls.solve(A, b, tau)


Solving using FOCUSS algorithm (only matrix based dictionaries supported)::

    import cr.sparse.cvx.focuss as focuss
    sol = focuss.matrix_solve_noiseless(A, b)


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
