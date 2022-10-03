# Copyright 2021 CR-Suite Development Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax.numpy as jnp

from .impl import _hermitian
from .lop import Operator
from .util import apply_along_axis
from jax.experimental.sparse import sparsify

###########################################################################################
#  Basic operators
###########################################################################################

def real_matrix(A):
    """Converts a real matrix into a linear operator

    Args:
        A (jax.numpy.ndarray): A real valued matrix (2D array) 

    Returns:
        Operator: A linear operator wrapping the matrix

    Forward operation: 

    .. math::

        y  = A x

    Adjoint operation:

    .. math::

        y = A^T x
    """
    m, n = A.shape
    def times(x):
        assert x.ndim == 1
        return A @ x
    def trans(x):
        assert x.ndim == 1
        return x @ A
    return Operator(times=times, trans=trans, shape=(m,n))

def matrix(A, axis=0):
    """Converts a matrix into a linear operator

    Args:
        A (jax.numpy.ndarray): A real or complex matrix (2D array) 
        axis (int): For multi-dimensional array input, the axis along which
          the linear operator will be applied 

    Returns:
        Operator: A linear operator wrapping the matrix

    Forward operation: 

    .. math::

        y  = A x

    Adjoint operation:

    .. math::

        y = A^H x = (x^H A)^H

    """
    m, n = A.shape
    real = jnp.isrealobj(A)

    times1d = lambda x: A @ x
    trans1d = lambda x : _hermitian(_hermitian(x) @ A )

    def times(x):
        """Forward matrix multiplication
        """
        if x.ndim == 1:
            return A @ x
        if x.ndim == 2:
            if axis == 0:
                return A @ x
            else:
                return x @ A.T
        # general case
        return jnp.apply_along_axis(times1d, axis, x)

    def trans(x):
        """Adjoint matrix multiplication
        """
        if x.ndim == 1:
            return trans1d(x)
        if x.ndim == 2:
            if axis == 0:
                return _hermitian(A) @ x
            else:
                return x @ jnp.conjugate(A)
        # general case
        return jnp.apply_along_axis(trans1d, axis, x)

    return Operator(times=times, trans=trans, shape=(m,n), real=real)


def sparse_real_matrix(A, axis=0):
    """Converts a sparse real matrix into a linear operator

    Args:
        A (jax.experimental.sparse.BCOO): A real valued sparse matrix in BCOO format 

    Returns:
        Operator: A linear operator wrapping the matrix

    Forward operation: 

    .. math::

        y  = A x

    Adjoint operation:

    .. math::

        y = A^T x
    """
    m, n = A.shape

    @sparsify
    def times1d(x):
        return A @ x

    @sparsify
    def trans1d(x):
        return x @ A
        

    @sparsify
    def times(x):
        """Forward matrix multiplication
        """
        if x.ndim == 1:
            return A @ x
        if x.ndim == 2:
            if axis == 0:
                return A @ x
            else:
                return x @ A.T
        # general case
        return jnp.apply_along_axis(times1d, axis, x)

    @sparsify
    def trans(x):
        """Adjoint matrix multiplication
        """
        if x.ndim == 1:
            return trans1d(x)
        if x.ndim == 2:
            if axis == 0:
                return A.T @ x
            else:
                return x @ A
        # general case
        return jnp.apply_along_axis(trans1d, axis, x)
    return Operator(times=times, trans=trans, shape=(m,n))


def diagonal(d, axis=0):
    """Returns a linear operator which mimics multiplication by a diagonal matrix

    Args:
        d (jax.numpy.ndarray): A vector (1D array) of diagonal entries
        axis (int): For multi-dimensional array input, the axis along which
          the linear operator will be applied 

    Returns:
        Operator: A linear operator wrapping the diagonal matrix multiplication
    """
    assert d.ndim == 1
    n = d.shape[0]
    times = lambda x: d * x
    trans = lambda x: _hermitian(d) * x
    times, trans = apply_along_axis(times, trans, axis)
    return Operator(times=times, trans=trans, shape=(n,n))


def scalar_mult(alpha, n):
    r"""Returns a linear operator T such that :math:`T v = \alpha v`

    Args:
        alpha (float): A scalar value
        n (int): The dimension of model/data space 

    Returns:
        Operator: A linear operator wrapping the scalar multiplication
    """
    alpha = jnp.asarray(alpha)
    alpha_c = jnp.conjugate(alpha)
    assert alpha.ndim == 0, "alpha must be a scalar quantity"
    times = lambda x: alpha * x
    trans = lambda x: alpha_c * x
    return Operator(times=times, trans=trans, shape=(n,n))


def zero(in_dim, out_dim=None, axis=0):
    """Returns a linear operator which maps everything to 0 vector in data space

    Args:
        in_dim (int): Dimension of the model space 
        out_dim (int): Dimension of the data space (default in_dim)
        axis (int): For multi-dimensional array input, the axis along which
          the linear operator will be applied 

    Returns:
        Operator: A zero linear operator
    """
    out_dim = in_dim if out_dim is None else out_dim
    times = lambda x: jnp.zeros(out_dim, dtype=x.dtype)
    trans = lambda x: jnp.zeros(in_dim, dtype=x.dtype)
    times, trans = apply_along_axis(times, trans, axis)
    return Operator(times=times, trans=trans, shape=(out_dim,in_dim))

def flipud(n):
    """Returns an operator which flips the order of entries in input upside down"""
    times = lambda x: jnp.flipud(x)
    trans = lambda x: jnp.flipud(x)
    return Operator(times=times, trans=trans, shape=(n,n))


def sum(n):
    """Returns an operator which computes the sum of a vector"""
    times = lambda x: jnp.sum(x, keepdims=True, axis=0)
    trans = lambda x: jnp.repeat(x, n, axis=0)
    return Operator(times=times, trans=trans, shape=(1,n))

def pad_zeros(n, before, after):
    """Adds zeros before and after a vector.

    Note:
        This operator is not JIT compliant
    """
    pad_1_dim = (before, after)
    pad_2_dim = ((before, after), (0, 0))
    m = before + n + after
    def times(x):
            return jnp.pad(x, pad_1_dim)
    def trans(x):
            return x[before:before+n]
    return Operator(times=times, trans=trans, shape=(m,n), matrix_safe=False)


def real(n):
    """Returns the real parts of a vector of complex numbers

    Note:
        This is a self-adjoint operator. 
        This is not a linear operator.
    """
    times = lambda x: jnp.real(x)
    trans = lambda x: jnp.real(x)
    return Operator(times=times, trans=trans, shape=(n,n), linear=False)


def symmetrize(n):
    """An operator which constructs a symmetric vector by pre-pending the input in reversed order
    """
    times = lambda x: jnp.concatenate((jnp.flipud(x), x))
    trans = lambda x: x[n:] + x[n-1::-1]
    return Operator(times=times, trans=trans, shape=(2*n,n))


def restriction(n, I, axis=0):
    """An operator which computes y = x[I] over an index set I

    Args:
        n (int): Dimension of model space
        I (jax.numpy.ndarray): 
        axis (int): For multi-dimensional array input, the axis along which
          the linear operator will be applied 
    """
    k = len(I)
    times1d = lambda x: x[I]
    trans1d = lambda x: jnp.zeros((n,), dtype=x.dtype).at[I].set(x)

    def times(x):
        if x.ndim == 1:
            return times1d(x)
        if x.ndim == 2:
            if axis == 0:
                # we apply column wise
                return x[I, :]
            # we apply row-wise
            return x[:, I]
        # general case
        return jnp.apply_along_axis(times1d, axis, x)

    def trans(x):
        if x.ndim == 1:
            return trans1d(x)
        # general case
        return jnp.apply_along_axis(trans1d, axis, x)

    return Operator(times=times, trans=trans, shape=(k,n))

def heaviside(n, axis=0, normalized=True):
    """Returns a linear operator implements the Heaviside step function

    Args:
        n (int): Dimension of the model space 
        axis (int): For multi-dimensional array input, the axis along which
          the linear operator will be applied
        normalized: If False, then simple cumsum, otherwise, apply on weighted x 

    Returns:
        Operator: A Heaviside linear operator

    Heaviside function is also known as the step function.
    In discrete domain, it is implemented as a cumulative sum
    operation. 

    An n x n Heaviside matrix has ones below and on
    the diagonal and zeros elsewhere.
    """
    w = jnp.sqrt(jnp.arange(n, 0, -1))
    wi = 1/w

    times_u = lambda x: jnp.cumsum(x)

    def trans_u(x):
        y = jnp.cumsum(x)
        ym = y[-1]
        return jnp.insert(ym - y[:-1], 0, ym)

    times_n = lambda x: jnp.cumsum(x * wi)

    def trans_n(x):
        y = jnp.cumsum(x)
        ym = y[-1]
        return jnp.insert(ym - y[:-1], 0, ym) * wi
    
    times, trans = (times_n, trans_n) if normalized else (times_u, trans_u)    
    times, trans = apply_along_axis(times, trans, axis)
    return Operator(times=times, trans=trans, shape=(n, n))

def cumsum(n, axis=0):
    return heaviside(n, axis, normalized=False)

def inv_heaviside(n, axis=0, normalized=True):
    """Returns a linear operator that computes the inverse of Heaviside/cumsum on input

    Args:
        n (int): Dimension of the model space 
        axis (int): For multi-dimensional array input, the axis along which
          the linear operator will be applied
        normalized(bool): Indicates if the Heaviside operator was normalized 

    Returns:
        Operator: An inverse of Heaviside linear operator

    Recall that Heaviside operate computes the cumulative sum.
    This operator computes the reverse of cumulative sum which
    is the difference of consecutive values.
    """
    w = jnp.sqrt(jnp.arange(n, 0, -1))

    times_u = lambda x: jnp.diff(x, prepend=0)
    trans_u = lambda x: -jnp.diff(x, append=0)

    times_n = lambda x: jnp.diff(x, prepend=0) * w
    trans_n = lambda x: -jnp.diff(x * w, append=0)

    times, trans = (times_n, trans_n) if normalized else (times_u, trans_u)    
    times, trans = apply_along_axis(times, trans, axis)
    return Operator(times=times, trans=trans, shape=(n, n))

def diff(n, axis=0):
    return inv_heaviside(n, axis, normalized=False)
