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

###########################################################################################
#  Basic operators
###########################################################################################

def identity(m, n=None):
    """Returns an identity linear operator from A to B"""
    n = m if n is None else n
    times = lambda x:  x
    trans = lambda x : x
    return Operator(times=times, trans=trans, shape=(m,n))

def matrix(A):
    """Converts a two-dimensional matrix to a linear operator"""
    m, n = A.shape
    times = lambda x: A @ x
    trans = lambda x : _hermitian(_hermitian(x) @ A )
    return Operator(times=times, trans=trans, shape=(m,n))

def diagonal(d):
    """Returns a linear operator which can be represented by a diagonal matrix"""
    assert d.ndim == 1
    n = d.shape[0]
    times = lambda x: d * x
    trans = lambda x: _hermitian(d) * x
    return Operator(times=times, trans=trans, shape=(n,n))


def zero(m,n=None):
    """Returns a linear operator which maps everything to 0 vector in data space"""
    n = m if n is None else n
    times = lambda x: jnp.zeros( (m,) + x.shape[1:], dtype=x.dtype)
    trans = lambda x: jnp.zeros((n,) + x.shape[1:], dtype=x.dtype)
    return Operator(times=times, trans=trans, shape=(m,n))

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
