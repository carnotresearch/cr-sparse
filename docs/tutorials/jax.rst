Thinking in JAX
========================


JAX API is similar to `NumPy`. However there are
many differences needed to achieve the Just In Time (JIT)
compilation of functions written in JAX. 
In this section, we give a set of examples showing
how to write numerical code properly with JAX.


Key points

* JAX arrays are immutable.
* It should be possible to statically determine
  the shape of function output variables from the
  shape of input variables for the JIT compiler.
* ``jax.lax`` is a low level module containing
  several helper functions to express complex
  logic in functional manner. We will often
  use its functions in the examples below.
* Standard Python ``for`` and ``while`` loops cannot be
  JIT compiled.
* A Python ``for`` loop will be 
  unrolled by JIT compiler (if the iteration count
  can be statically determined). This increases
  compilation time and should be avoided.
* JAX provides ``lax.while_loop`` and ``lax.fori_loop``
  as functional alternatives. 
* JAX doesn't support array views.
* If there are some arguments to a function
  which determine the shape of intermediate
  arrays in the function body or the output
  of the function body, then they must be
  marked via ``static_argnums`` or
  ``static_argnames`` to the JIT compiler.
* The common ``if else`` Python blocks cannot
  be JIT compiled. You can use ``lax.cond``
  or ``jnp.where`` for building equivalent
  logic.

Sometimes, you may worry that you are writing
too many low level functions just to make the
JIT compiler happy to implement 
some logic which could have been done by using
some for/while loops in normal NumPy code.
But this additional complexity pays off in the
end. If the JIT compiler accepts your implementation, it
will generate code which will usually be much
faster than NumPy version.

As we are not used to writing functional code,
it takes a lot of effort to come up with proper
JAX compatible designs in the beginning.
The more you code in JAX, the easier it becomes
to think functionally.  

All the code snippets in this tutorial
are taken from the code in ``CR-Nimble``
and ``CR-Sparse`` libraries.

In the following ``jnp`` is a short name
for ``jax.numpy`` module::

    import jax.numpy as jnp


.. rubric:: Activating 64-bit mode

By default, JAX uses 32-bit for floating point
numbers. For sparse reconstruction algorithms,
32-bit precision is often not enough.
Do make sure to configure JAX to use 64-bit
floating point numbers before calling any
JAX functions::

    from jax.config import config
    config.update("jax_enable_x64", True)


1D arrays
---------------

.. rubric:: Modifying vectors


Set a value at a particular index::

    x.at[index].set(value)

Add a value at a particular index::

    x.at[index].add(value)

Subtract a value at a particular index::

    x.at[index].add(-value)

Swapping two elements at index i and j::

    xi = x[i]
    xj = x[j]
    x = x.at[i].set(xj)
    x = x.at[j].set(xi)


.. rubric:: Simple checks

Check if a vector contains increasing values::

    jnp.all(jnp.diff(x) > 0)


Check if all values in a vector are equal::

    jnp.all(x == x[0])


.. rubric:: Basic manipulation

Convert a vector to a row vector (1xn)::

    jnp.expand_dims(x, 0)

Convert a vector to a column vector (nx1)::

    jnp.expand_dims(x, 1)


Construct a unit vector of length n with a zero
in i-th dimension::

    jnp.zeros(n).at[i].set(1)

Note that the length of the array given by
n has to be statically determined by the JIT
compiler.

Right shift the contents of a vector by one element::

    jnp.zeros_like(x).at[1:].set(x[:-1])

* We first construct an array of the same shape
  as x containing all zeros.
* We then fill the n -1 elements in this array
  (except the first element) with the first n-1
  elements of x.
* The last element of x is left out.
* Our focus is on expressing our logic in a functional
  manner.
* We leave it to the JIT compiler to come up with
  the efficient implementation of the logic for the
  target architecture.


If we want to right shift by n elements, then the logic
becomes::

    jnp.zeros_like(x).at[n:].set(x[:-n])


Return the magnitudes of elements of a vector
in descending order::

    jnp.sort(jnp.abs(x))[::-1]

* We first get the magnitudes ``jnp.abs(x)``
* We then sort the result using ``jnp.sort`` ascending order.
* We finally reverse the array in descending order by indexing
  ``[::-1]``.


Let us be more adventurous.
We wish to find out how many of the largest elements
in a vector ``a`` are enough to capture a fraction ``q``
of the total energy of the vector ``a``.
The vector can be real or complex. We shall break this
down into multiple steps.


Compute energy of individual elements::

    a = jnp.conj(a) * a

Sort the energies in descending order::

    a = jnp.sort(a)[::-1]

Compute the total energy::

    s = jnp.sum(a)

Normalize the energies to fractions::

    a = a / s

Compute the cumulative energies starting from the
largest coefficient::

    cmf = jnp.cumsum(a)

Find the index at which the cumulative energy reaches
the required fraction ``q``::

    index =  jnp.argmax(cmf >= q)

The required number of elements to capture ``q``
fraction of energy is ``index + 1``.



.. rubric:: Conditional code

Consider the following function::

    def f(x, alpha):
        if alpha == 0:
            return x
        return x / alpha

We shall now build this logic using ``lax.cond`` step by step.
The condition to check is ``alpha === 0``.

We have to define two functions. One for the case where
the condition is true and another for the case where the
condition is false. For both cases, we shall define
anonymous functions using the ``lambda`` keyword.

Here is the function for the true case::

    lambda x : x

Here is the function for the false case::

    lambda x: x / alpha

Both functions take ``x`` as argument. Now, we can combine
these elements to form our functional equivalent code::

    lax.cond(alpha == 0, lambda x : x, lambda x: x / alpha, x)

We suggest you to read the official documentation
to understand the details of ``lax.cond``.


.. rubric:: Circular buffers

A circular buffer is a fixed size array in which one
can push values either left or right side. When
we push a new element, an old element from the
other side is removed.

Assume that we are given a buffer ``buff`` and
need to push a value ``val`` from the left side::

    buf.at[1:].set(buf[:-1]).at[0].set(val)


If we need to push a value from the right side::

    buf.at[:-1].set(buf[1:]).at[-1].set(val)


Norms
----------------

``jnp.linalg.norm`` is the workhorse for
general norm computation. However, we
can often use simple computations for
specific cases ourselves.

Computing the l-1 norm::

    jnp.sum(jnp.abs(x))


Computing the l-2 norm::

    jnp.sqrt(jnp.abs(jnp.vdot(x, x)))


Computing the l-inf norm::

    jnp.max(jnp.abs(x))


.. rubric:: Column wise norms

Often in sparse signal processing, we
are dealing with a matrix consisting
of vectors arranged column wise
where we have to compute the norm of
each vector.

Column-wise l-2 norm::

    jnp.linalg.norm(X, ord=2, axis=0, keepdims=False)


The ``keepdims=False`` flag is needed to ensure
that the result is reduced to a 1D array.

If we wish to compute the norm along rows, we can just
change ``axis=1``.


A common task is normalizing a vector so that it becomes
unit norm. Care must be taken for the case where the
vector is zero.

We can shift the norm value by a very small amount
before carrying out the division. For 32-bit
floating point numbers, the smallest positive value
is given by::

    EPS = jnp.finfo(jnp.float32).eps

Then normalization can be written as::

    s = jnp.sqrt(jnp.abs(jnp.vdot(x, x))) + EPS
    x = jnp.divide(x, s)

This approach avoids a conditional expression
using ``lax.cond``. It is good to avoid
conditional code as much as possible as
they become bottlenecks (especially when the
numerical code is running on GPU hardware).
Since this normalization is slightly
inaccurate, you should examine the use case
if this inaccuracy is acceptable or not.


Matrices
-----------------


Checking if a matrix is symmetric::

    jnp.array_equal(A, A.T)

Computing the Hermitian transpose::

    jnp.conjugate(jnp.swapaxes(A, -1, -2))


Checking if a real matrix has orthogonal columns::

    G = A.T @ A
    m = G.shape[0]
    I = jnp.eye(m)
    result = jnp.allclose(G, I)


Checking for orthogonal rows::

    G = A @ A.T
    m = G.shape[0]
    I = jnp.eye(m)
    result = jnp.allclose(G, I, atol=m*m*atol)

Extracting the off-diagonal elements of a matrix::

    mask = ~jnp.eye(*A.shape, dtype=bool)
    off_diagonal_elements = A[mask]


Setting the diagonal elements of a given matrix::

    indices = jnp.diag_indices(A.shape[0])
    A = A.at[indices].set(value)

Adding something to the diagonal elements of a matrix::

    indices = jnp.diag_indices(A.shape[0])
    A = A.at[indices].add(value)


Finding the index of the largest element (by magnitude)
in each column of a matrix::

    jnp.argmax(jnp.abs(A), axis=0)


Premultiplying a matrix A with a diagonal matrix
whose diagonal elements are given by a vector d::

    jnp.multiply(d[:, None], A)

Post-multiplying a matrix A with a diagonal matrix
whose diagonal elements are given by a vector d::

    jnp.multiply(A, d)

Extracting bxb diagonal blocks from a matrix::

    n = A.shape[0]
    nb = n // b
    starts = [i*b for i in range(nb)]
    blocks = jnp.array([A[s:s+b,s:s+b] for s in starts])



Linear Algebra
---------------------


.. rubric:: Constructing a Toeplitz matrix

A Toeplitz matrix is completely specified by
its first row and column. E.g., ::

    [[1 2 3 4]
     [2 1 2 3]
     [3 2 1 2]
     [4 3 2 1]]


Suppose we are given the first row and first column
of the Toeplitz matrix and we are required to
construct the whole matrix. We can do so in a
fashion which doesn't require any loops.
It is achieved by indexing magic.
::


    def toeplitz_mat(c, r):
        m = len(c)
        n = len(r)
        # assert c[0] == r[0]
        w = jnp.concatenate((c[::-1], r[1:]))
        # backwards indices
        a = -jnp.arange(m, dtype=int)
        # forwards indices
        b = jnp.arange(m-1,m+n-1, dtype=int)
        # combine indices for the toeplitz matrix
        indices = a[:, None] + b[None, :]
        # form the toeplitz matrix
        mat = w[indices]
        return mat

We combined the first row and first column
elements into a single array w. Then constructed
an index matrix where each element in the index
matrix is an index in the w array identifying the
element to be placed in the output Toeplitz matrix.
Forming the Toeplitz matrix then becomes a simple
indexing step.


Basic Signal Processing
---------------------------------


Scaling a vector to the range 0 and 1::

    shift = jnp.min(x)
    x = x - shift
    scale = jnp.max(x)
    x = x / scale

Reverting back::

    x = x * scale
    x = x + shift


Hard thresholding to K largest elements::

    indices = jnp.argsort(jnp.abs(x))
    I = indices[:-K-1:-1]
    x_I = x[I]

Here the tuple of ``(I, x_I)`` identifies
the indices and values of K largest entries.
To build the full length approximation, we will have
to do the following::

    x = jnp.zeros_like(x)
    x = x.at[I].set(x_I)


Alternatively, we can do the following
to compute the K sparse approximation::

    indices = jnp.argsort(jnp.abs(x))
    x = x.at[indices[:-K]].set(0)


.. rubric:: Sliding windows

A common signal processing task is to
divide a signal x into windows of length
w each such that consecutive windows
have an overlap of m samples.
Achieving this in JAX will require some indexing trick
again::

    step = w - m
    starts = jnp.arange(0, len(x) - w + 1, step)
    block = jnp.arange(w)
    idx = starts[:, None] + block[None, :]
    windows = x[idx]

This constructs the windows of x in each row
of the resulting matrix. If you wish the
windows to be column wise, just take the
transpose.




