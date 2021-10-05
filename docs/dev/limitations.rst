Limitations
======================


This section is a rough summary of some of the limitations 
encountered during the development of this package. 
Be aware that some of these limitations may be entirely 
due to my limited knowledge of JAX library. Hopefully, 
in future, with my better knowledge, or with better support
from JAX, some of these limitations will be alleviated.


.. rubric:: Key issues with JAX 

This is a list of key issues with JAX library which have
been acknowledged with JAX development team also.

* Lack of support for dynamic or data dependent shapes in JAX.
  See `#8042 <https://github.com/google/jax/discussions/8042>`_
* 1D convolution is slow in JAX (especially on CPU). 
  See `#7961 <https://github.com/google/jax/discussions/7961>`_.



Data clustering
-----------------------

- Spectral clustering assess the number of clusters 
  from data. It cannot be jitted. In turn, the k-means
  invocation from inside spectral clustering cannot be jitted either.