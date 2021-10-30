# Copyright 2021 CR.Sparse Development Team
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
    times = lambda x: A @ x
    trans = lambda x : _hermitian(_hermitian(x) @ A )
    times, trans = apply_along_axis(times, trans, axis)
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


def restriction(n, indices):
    """An operator which computes y = x[I] over an index set I
    """
    k = len(indices)
    times = lambda x: x[indices]
    trans = lambda x: jnp.zeros((n,)+x.shape[1:]).at[indices].set(x)
    return Operator(times=times, trans=trans, shape=(k,n))
