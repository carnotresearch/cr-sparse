Limitations
======================


This section is a rough summary of some of the limitations 
encountered during the development of this package. 
Some of these limitations are due to not enough time 
spent on development. Others are due to lack of support
in JAX library.
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
* Support for sparse matrices is still under development in JAX.


Utilities
----------------------

- off_diagonal_elements cannot be jitted. 
  JAX arrays do not support boolean scalar indices.

Data clustering
-----------------------

- Spectral clustering assess the number of clusters 
  from data. It cannot be jitted. In turn, the k-means
  invocation from inside spectral clustering cannot be jitted either.
- Normalized spectral clustering could make use of 
  sparse matrices (identity and diagonal) whenever
  it becomes available.

Signal Processing
----------------------

- Walsh Hadamard transform implementation is technically correct 
  but possibly not vectorized enough. It is slow.

Linear Operators
---------------------

- Some operators like pad don't work well on matrix input yet.