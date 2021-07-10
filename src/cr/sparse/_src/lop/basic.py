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
from .lop import LinearOperator

###########################################################################################
#  Basic operators
###########################################################################################

def identity(m, n):
    """Returns an identity linear operator from A to B"""
    times = lambda x:  x
    trans = lambda x : x
    return LinearOperator(times=times, trans=trans, m=m, n=n)

def matrix(A):
    """Converts a two-dimensional matrix to a linear operator"""
    m, n = A.shape
    times = lambda x: A @ x
    trans = lambda x : _hermitian(_hermitian(x) @ A )
    return LinearOperator(times=times, trans=trans, m=m, n=n)

def diagonal(d):
    """Returns a linear operator which can be represented by a diagonal matrix"""
    assert d.ndim == 1
    n = d.shape[0]
    times = lambda x: d * x
    trans = lambda x: _hermitian(d) * x
    return LinearOperator(times=times, trans=trans, m=n, n=n)


def zero(m,n=None):
    """Returns a linear operator which maps everything to 0 vector in data space"""
    n = m if n is None else n
    times = lambda x: jnp.zeros( (m,) + x.shape[1:] )
    trans = lambda x: jnp.zeros((n,) + x.shape[1:])
    return LinearOperator(times=times, trans=trans, m=n, n=n)

def flipud(n):
    """Returns an operator which flips the order of entries in input upside down"""
    times = lambda x: jnp.flipud(x)
    trans = lambda x: jnp.flipud(x)
    return LinearOperator(times=times, trans=trans, m=n, n=n)
