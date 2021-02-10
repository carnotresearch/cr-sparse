The Beautiful Einstein Summations
=======================================

.. contents::
    :depth: 3
    :local:

.. highlight:: python

`Einstein summations <https://en.wikipedia.org/wiki/Einstein_notation>`_
are a powerful mechanism for array operations.
Major scientific computing / neural network
libraries like 
`numpy <https://numpy.org/doc/stable/reference/generated/numpy.einsum.html>`_,
`TensorFlow <https://www.tensorflow.org/api_docs/python/tf/einsum>`_,
and `PyTorch <https://pytorch.org/docs/stable/generated/torch.einsum.html>`_ have built in support for
performing linear algebraic array operations using Einstein 
summation convention. 
This article is a gentle introduction on using 
Einstein summations for such operations. The libraries
have highly efficient implementations available for
computing Einstein summations and thus they would run
much faster than any hand written alternative code. 


.. rubric:: Notation

Vectors will be denoted using small letters
:math:`u, v, w, x, y, z`. Scalars will also be small letters
:math:`a,b,c,m,n,s` etc.
The letters :math:`i,j,k,l` will be used to indicate indices.
Matrices will be denoted with capital letters :math:`A,B,C`
etc.

Elements of a vector will be denoted using ``numpy``
notation :math:`v[i]`.
Elements, rows and columns of a matrix will also be denoted 
using ``numpy`` notation :math:`A[i,j]`, 
:math:`A[i,:]` and :math:`A[:, j]`.

Einstein Notation
---------------------------

Consider a matrix multiplication operation :math:`C = A B`.
Individual elements of :math:`C` are computed by the summation:

.. math::

    C[i, k] = \sum_{j} A[i, j] B [j, k]

Einstein notation simply drops the summations from the equation:    

.. math::

    C[i, k] = A[i, j] B [j, k]

The idea is simple. If an index doesn't appear in the
L.H.S. but it appears in the R.H.S., then we assume that
the expression on the R.H.S. is to be summed over that 
index. In the matrix multiplication example above, since 
the index :math:`j` appears on R.H.S. and doesn't appear
on the L.H.S., we assume that the expression 
:math:`A[i, j] B [j, k]` is to be summed over :math:`j`.

The equation :math:`s = v[i]` would mean that we need
to sum over all the elements of a vector :math:`v`.

The equation :math:`s = A[i,j]` would mean that we 
need to sum over all the elements of the matrix :math:`A`.


The equation :math:`C[i, j] = A [i, j] B [i, j]` would
mean that we need to multiply :math:`A` and :math:`B` element-wise
to compute :math:`C`.

The equation :math:`v[i] = A[i, j]` would mean that each
element of the vector :math:`v` is the sum of elements in
corresponding row of matrix :math:`A`, i.e. row wise sum of :math:`A`.

In this article, we will explore Einstein summations
using ``numpy``.


``einsum``
---------------

``numpy.einsum`` is a general purpose workhorse 
function in ``numpy`` for computing summations following
the Einstein Notation a.k.a. Einstein summations. 
For example, the matrix multiplication :math:`C = AB`
is computed as ``einsum('ij,jk->ik', A, B)``. 

.. note::

    The explanation may seem heavy in first go. Read it once,
    read all the examples in following sections, come back,
    and read this section again.

To understand this function, we need to understand
the subscripts specification in the first argument
``ij,jk->ik``. Let's revisit the matrix multiplication equation:

.. math::

    C[i, k] = A [i, j] B [j, k]

* There are two arguments to matrix multiplication: :math:`A` and :math:`B`.
* :math:`A` is indexed by :math:`i` (for rows) and :math:`j` (for columns).
* :math:`B` is indexed by :math:`j` (for rows) and :math:`k` (for columns).
* The output of the operation is the matrix :math:`C`.
* :math:`C` is indexed by :math:`i` (for rows) and :math:`k` (for columns).

Now let's parse ``ij,jk->ik``.

* The Einstein summation ``einsum('ij,jk->ik', A, B)`` has two arguments :math:`A`
  and :math:`B`.
* The specification ``ij,jk->ik`` describes how the summation is to proceed.
* There are two parts: ``ij,jk`` and ``ik`` separated by ``->``. Input -> Output.
* ``ik`` are the indices for the output of the summation
* ``ij,jk`` are indices for the input arguments for the summation
* The comma separates indices for different input arguments. 
  Thus, in this case, there are two arguments.
* Splitting ``ij,jk``, ``ij`` are indices for first argument :math:`A` and
  ``jk`` are indices for second argument :math:`B`. 
* Since there are two letters for each argument, we assume that the
  arguments are both two dimension arrays (i.e. matrices).
* The first letter refers to the index for the rows of a matrix and the second 
  letter refers to the index for the columns of the same matrix.
* Thus, it translates to ``A[i, j] * B[j, k]`` and the result is stored in
  ``C[i,k]``.

We will now look at all kinds of linear algebraic array operations possible
via ``einsum``.

Load the ``numpy`` library first::

    >>> import numpy as np

In the examples below:

* We present the underlying mathematical equation first in Einstein notation
  for each linear algebraic operation.
* We then present its implementation using ``einsum``.
* We also contrast it with the standard ``numpy`` functions for same operation.

Vector Operations
---------------------------

Let :math:`u, v` be a vectors of length :math:`n=3`.


::

    >>> u = np.array([0, 1, 4])
    >>> v = np.array([1, 2, 3])


Sum of all elements of a vector
''''''''''''''''''''''''''''''''''''''''''''''''''

In normal mathematical notation, we say :math:`s = \sum_{i=0}^{n-1} v[i]`. 
In Einstein notation, we say :math:`s = v[i]`.

::

    >>> np.einsum("i->", u)
    5
    >>> np.einsum("i->", v)
    6

Element wise multiplication
''''''''''''''''''''''''''''''''''''''''''''''''''

.. math::

    w[i] = u[i] v[i]

In this case, there is no summation happening in the R.H.S.

::

    >>> w = np.einsum("i,i->i", u, v)
    >>> w
    array([ 0,  2, 12])


Inner Product
''''''''''''''''''''''''''''''''''''''''''''''''''

Normal mathematical notation:

.. math::

    s = \langle u, v \rangle = \sum_{i=0}^{n-1} u[i] v[i]


Einstein notation:

.. math::

    s = \langle u, v \rangle = u[i] v[i]

::

    >>> np.einsum("i,i->", u, v)
    14

    >>> np.inner(u, v)
    14

    >>> np.dot(u, v)
    14


Outer Product
''''''''''''''''''''''''''''''''''''''''''''''''''

.. math::

    A[i,j] = u[i] v[j]

::

    >>> np.outer(u, v)
    array([[ 0,  0,  0],
       [ 1,  2,  3],
       [ 4,  8, 12]])


    >>> np.einsum('i,j->ij', u, v)
    array([[ 0,  0,  0],
       [ 1,  2,  3],
       [ 4,  8, 12]])



Matrix Operations
-------------------------

Let's construct 2 square matrices :math:`A, B` for the examples below.

::

    >>> A = np.array([[0,1,2],[3,4,5],[0,1,2]])
    >>> B = np.array([[-1,1,-2],[3,2,-3],[0,-1,1]])
    >>> A
    array([[0, 1, 2],
       [3, 4, 5],
       [0, 1, 2]])
    >>> B
    array([[-1,  1, -2],
       [ 3,  2, -3],
       [ 0, -1,  1]])

Matrix Transpose
''''''''''''''''''''''''''''''''''''''''''''''''''

The expression :math:`C  = A^T` is computed as:

.. math::

    C[j, i] = A [i, j]

::

    >>> np.einsum('ij->ji', A)
    array([[0, 3, 0],
       [1, 4, 1],
       [2, 5, 2]])

    >> A.T
    array([[0, 3, 0],
       [1, 4, 1],
       [2, 5, 2]])

Main Diagonal
''''''''''''''''''''''''''''''''''''''''''''''''''

The main diagonal of a matrix :math:`A` can be extracted as:

.. math::

    v[i] = A[i,i]

::

    >>> np.einsum('ii->i', A)
    array([0, 4, 2])

    >>> np.diag(A)
    array([0, 4, 2])

Trace
''''''''''''''''''''''''''''''''''''''''''''''''''

The trace of a matrix is the sum of its diagonal elements.

.. math::
    
    \text{Trace}(A) = \sum_{i} A[i, i]

In Einstein notation:

.. math::

    \text{Trace}(A) = A[i, i]


::

    >>> np.einsum('ii->', A)
    6


Sum of Elements
''''''''''''''''''''''''''''''''''''''''''''''''''

The sum of all elements of a matrix can be expressed as
:math:`\sum_{i} \sum_{j} A[i, j]`. In Einstein notation:

.. math::

    s = A[i, j]

:: 

    >>> np.einsum('ij->', A)
    18


Column-wise Sum
''''''''''''''''''''''''''''''''''''''''''''''''''

.. math::

    v[j] = A[i, j]  = \sum_i A[i, j]

::

    >>> np.sum(A, axis=0)
    array([3, 6, 9])

    >>> np.einsum('ij->j', A)
    array([3, 6, 9])

Row-wise Sum
''''''''''''''''''''''''''''''''''''''''''''''''''

.. math::

    v[i] = A[i, j] = \sum_j A[i, j]

:: 

    >>> np.sum(A, axis=1)
    array([ 3, 12,  3])

    >>> np.einsum('ij->i', A)
    array([ 3, 12,  3])


Matrix Multiplication
''''''''''''''''''''''''''''''''''''''''''''''''''

We can consider four possibilities:

#. :math:`C = A B`
#. :math:`C = A^T B`
#. :math:`C = A B^T`
#. :math:`C = A^T B^T = (BA)^T`


.. rubric:: :math:`C = A B`

.. math::

    C[i, k] = A[i, j] B [j, k]

::

    >>> A @ B
    array([[  3,   0,  -1],
       [  9,   6, -13],
       [  3,   0,  -1]])

    >>> np.einsum('ij,jk->ik', A, B)
    array([[  3,   0,  -1],
       [  9,   6, -13],
       [  3,   0,  -1]])


.. rubric:: :math:`C = A^T B`

.. math::

    C[i, k] = A[j, i] B [j, k]

::

    >>> A.T @ B
    array([[  9,   6,  -9],
       [ 11,   8, -13],
       [ 13,  10, -17]])

    >>> np.einsum('ji,jk->ik', A, B)
    array([[  9,   6,  -9],
       [ 11,   8, -13],
       [ 13,  10, -17]])


.. rubric:: :math:`C = A B^T` (inner product)

.. math::

    C[i, k] = A[i, j] B [k, j]

::

    >>> A @ B.T
    array([[-3, -4,  1],
       [-9,  2,  1],
       [-3, -4,  1]])

    >>> np.einsum('ij,kj->ik', A, B)
    array([[-3, -4,  1],
       [-9,  2,  1],
       [-3, -4,  1]])

This is also known as inner product of two matrices
in ``numpy`` as ``numpy`` is row major.

::

    >>> np.inner(A, B)
    array([[-3, -4,  1],
       [-9,  2,  1],
       [-3, -4,  1]])


.. rubric:: :math:`C = A^T B^T`

.. math::

    C[i, k] = A[j, i] B [k, j]

::

    >>> A.T @ B.T
    array([[ 3,  6, -3],
       [ 1,  8, -3],
       [-1, 10, -3]])

    >>> np.einsum('ji,kj->ik', A, B)
    array([[ 3,  6, -3],
       [ 1,  8, -3],
       [-1, 10, -3]])

Element-wise Multiplication
''''''''''''''''''''''''''''''''''''''''''''''''''

.. rubric:: :math:`C[i,j] = A[i, j] B[i,j]`

::

    >>> A * B
    array([[  0,   1,  -4],
       [  9,   8, -15],
       [  0,  -1,   2]])

    >>> np.einsum('ij,ij->ij', A, B)
    array([[  0,   1,  -4],
       [  9,   8, -15],
       [  0,  -1,   2]])


We can similarly construct element wise multiplications
with the transposes of either A or B or both.

.. rubric:: :math:`C[i, j] = A[i,j] B[j, i]`

::

    >>> A * B.T
    array([[ 0,  3,  0],
       [ 3,  8, -5],
       [ 0, -3,  2]])

    >>> np.einsum('ij,ji->ij', A, B)
    array([[ 0,  3,  0],
       [ 3,  8, -5],
       [ 0, -3,  2]])


Column-wise Dot Product
''''''''''''''''''''''''''''''''''''''''''''''''''

Sometimes we need to compute the inner product
of each column of :math:`A` with corresponding
column of :math:`B`. 

Suppose we have :math:`n` columns in both :math:`A`
and :math:`B` with :math:`m` rows each. Then
the result will be a vector of dimension :math:`n`
such that:

.. math::
    
    v[j] = \langle A[:, j], B[:, j] \rangle

In a way we are multiplying the matrices element wise
and then summing over each column. In Einstein notation:

.. math::

    v[j] = A[i, j] B[i, j]

::

     >>> np.einsum('ij,ij->j', A, B)
     array([  9,   8, -17])

     >>> np.sum(A*B, axis=0)
     array([  9,   8, -17])

General Array Operations
--------------------------------

Following operations work on arrays of any dimension.

A view of an array::

    >>> np.einsum('...', A)
    array([[0, 1, 2],
       [3, 4, 5],
       [0, 1, 2]])



