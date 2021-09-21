.. _api:wavelets:

Wavelets
=====================

.. contents::
    :depth: 2
    :local:


``CR-Sparse`` provides support for both DWT (Discrete Wavelet Transform)
and CWT (Continuous Wavelet Transform).


The support for discrete wavelets is a partial port of 
wavelets functionality from the `PyWavelets <https://pywavelets.readthedocs.io/en/latest/>`_ project. 
The functionality has been written on top of JAX. While PyWavelets gets its performance from the C extensions in its 
implementation, we have built the functionality on top of JAX API. 

- API and implementation are both based on functional programming paradigm.
- There are no C extensions. All implementation is pure Python.
- The implementation takes advantage of XLA and can run easily on GPUs and TPUs.

Continuous Wavelet Transform have been implemented following :cite:`torrence1998practical`.
A reference implementation using NumPy is 
`here <https://github.com/aaren/wavelets>`_. 


The code examples in this section will assume following imports::

  import cr.sparse as crs
  import cr.sparse.wt as wt


Discrete Wavelets
--------------------------------------


API is available at two levels

- Functions which directly correspond to the high level API of PyWavelets.
- Lower level functions which are JIT compiled.

The high level functions involve handling a variety of use cases for
the arguments passed. For example, they can accept lists as well as
JAX nd-arrays. These functions cannot be JIT compiled. Lower level 
functions have been carefully designed to accept arguments which 
fit the JIT rules of JAX. They can be embedded in another JIT 
compiled function.

Current support is focused on discrete wavelet transforms. 
Following wavelets are supported.

    bior1.1 bior1.3 bior1.5 bior2.2 bior2.4 bior2.6 bior2.8 bior3.1 bior3.3 bior3.5 bior3.7 bior3.9 bior4.4 bior5.5 bior6.8 coif1 coif2 coif3 coif4 coif5 coif6 coif7 coif8 coif9 coif10 coif11 coif12 coif13 coif14 coif15 coif16 coif17 db1 db2 db3 db4 db5 db6 db7 db8 db9 db10 db11 db12 db13 db14 db15 db16 db17 db18 db19 db20 db21 db22 db23 db24 db25 db26 db27 db28 db29 db30 db31 db32 db33 db34 db35 db36 db37 db38 dmey haar rbio1.1 rbio1.3 rbio1.5 rbio2.2 rbio2.4 rbio2.6 rbio2.8 rbio3.1 rbio3.3 rbio3.5 rbio3.7 rbio3.9 rbio4.4 rbio5.5 rbio6.8 sym2 sym3 sym4 sym5 sym6 sym7 sym8 sym9 sym10 sym11 sym12 sym13 sym14 sym15 sym16 sym17 sym18 sym19 sym20


.. currentmodule:: cr.sparse.wt


High-level API
'''''''''''''''''''''''''''''''

.. rubric:: Data types

.. autosummary::
  :nosignatures:
  :toctree: _autosummary
  :template: namedtuple.rst

    FAMILY
    SYMMETRY

.. autosummary::
  :nosignatures:
  :toctree: _autosummary

    DiscreteWavelet

.. rubric:: Wavelets

.. autosummary::
  :toctree: _autosummary

  families
  build_wavelet
  wavelist
  is_discrete_wavelet
  wname_to_family_order
  build_discrete_wavelet



.. rubric:: Wavelet transforms

.. autosummary::
  :toctree: _autosummary

    dwt
    idwt
    dwt2
    idwt2
    downcoef
    upcoef
    wavedec
    waverec
    dwt_axis
    idwt_axis
    dwt_column
    dwt_row
    dwt_tube
    idwt_column
    idwt_row
    idwt_tube

.. rubric:: Utilities

.. autosummary::
  :toctree: _autosummary

    modes
    pad
    dwt_max_level
    dwt_coeff_len
    up_sample

Lower-level API
'''''''''''''''''''''''''''''


.. autosummary::
  :toctree: _autosummary

  dwt_
  idwt_ 
  downcoef_
  upcoef_
  dwt_axis_
  idwt_axis_


.. _ref-wt-modes:

Signal Extension Modes
''''''''''''''''''''''''''''''''

Real world signals are finite. They are typically stored in 
finite size arrays in computers. Computing the wavelet transform
of signal values around the boundary of the signal inevitably involves
assuming some form of signal extrapolation. A simple extrapolation
method is to extend the signal with zeros at the boundary. 
Reconstruction of the signal from its wavelet coefficients may introduce
boundary artifacts based on how the signal was extrapolated. A careful
choice of signal extension method is necessary based on actual 
application. 

We provide following signal extension modes at the moment.

zero
  Signal is extended by adding zeros::

    >>> wt.pad(jnp.array([1,2,4,-1,2,-1]), 2, 'zero')
    DeviceArray([ 0,  0,  1,  2,  4, -1,  2, -1,  0,  0], dtype=int64)


constant
  Border values of the signal are replicated::

    >>> wt.pad(jnp.array([1,2,4,-1,2,-1]), 2, 'constant')
    DeviceArray([ 1,  1,  1,  2,  4, -1,  2, -1, -1, -1], dtype=int64)


symmetric
  Signal is extended by mirroring the samples at the border in mirror form. 
  The border sample is also mirrored.::

    >>> wt.pad(jnp.array([1,2,4,-1,2,-1]), 2, 'symmetric')
    DeviceArray([ 2,  1,  1,  2,  4, -1,  2, -1, -1,  2], dtype=int64)


reflect
  Signal is extended by reflecting the samples around the border sample.
  Border sample is not copied in the extension.:: 

    >>> wt.pad(jnp.array([1,2,4,-1,2,-1]), 2, 'reflect')
    DeviceArray([ 4,  2,  1,  2,  4, -1,  2, -1,  2, -1], dtype=int64)

periodic
  Signal is extended periodically. The samples at the end repeat at the extension
  at the beginning. The samples at the beginning repeat at the extension at the end.::

    >>> wt.pad(jnp.array([1,2,4,-1,2,-1]), 2, 'periodic')
    DeviceArray([ 2, -1,  1,  2,  4, -1,  2, -1,  1,  2], dtype=int64)

periodization
  The signal is extended the same way as the periodic extension. The major difference is that
  the number of wavelet coefficients is identical to the length of the signal. All extra values
  are trimmed.

Many of the signal extension modes are similar to the padding modes supported by the
``jax.numpy.pad`` function. However, the naming convention is different and follows 
PyWavelets.


Continuous Wavelets
-----------------------------------


.. rubric:: Further Reading

.. bibliography::
   :filter: docname in docnames