Digital Signal Processing
===============================

.. contents::
    :depth: 2
    :local:

The ``CR-Sparse`` library has some handy digital signal processing routines
implemented in JAX. They are available as part of the ``cr.sparse.dsp``
package.


Synthetic Signals
-----------------------

.. currentmodule:: cr.sparse.dsp.signals

.. autosummary::
    :nosignatures:
    :toctree: _autosummary

    pulse
    transient_sine_wave
    decaying_sine_wave
    chirp
    chirp_centered
    gaussian_pulse


.. currentmodule:: cr.sparse.dsp



Discrete Cosine Transform
-------------------------------------

.. autosummary::
    :nosignatures:
    :toctree: _autosummary

    dct
    idct
    orthonormal_dct
    orthonormal_idct

.. currentmodule:: cr.sparse.dsp

Fast Walsh Hadamard Transform
------------------------------

There is no separate Inverse Fast Walsh Hadamard Transform as FWHT is the inverse of
itself except for a normalization factor.
In other words,  ``x == fwht(fwht(x)) / n`` where n is the length of x.

.. autosummary::
    :nosignatures:
    :toctree: _autosummary

    fwht

