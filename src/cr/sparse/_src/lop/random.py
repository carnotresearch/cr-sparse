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
import jax.random

from .impl import _hermitian
from .lop import Operator
from .basic import matrix as matrix_op
from .basic import sparse_real_matrix, real_matrix
import cr.sparse.dict as crdict
from .util import apply_along_axis


def gaussian_dict(key, m, n=None, normalize_atoms=True, axis=0):
    """An operator which represents a Gaussian sensix matrix (with normalized columns)

    Args:
        key: a PRNG key used as the random key.
        m (int): Number of rows of the sensing matrix 
        n (int): Number of columns of the sensing matrix
        normalize_atoms (bool): Whether the columns of sensing matrix are normalized 
          (default True)
        axis (int): For multi-dimensional array input, the axis along which
          the linear operator will be applied 

    Returns:
        Operator: A linear operator wrapping the sensing matrix

    Examples:
        - :ref:`gallery:lop:cs_operators`
    """
    n = m if n is None else n
    Phi = crdict.gaussian_mtx(key, m, n, normalize_atoms=normalize_atoms)
    return matrix_op(Phi, axis=axis)


def rademacher_dict(key, m, n=None, normalize_atoms=True, axis=0):
    """An operator which represents a Rademacher sensing matrix

    Args:
        key: a PRNG key used as the random key.
        m (int): Number of rows of the sensing matrix 
        n (int): Number of columns of the sensing matrix
        normalize_atoms (bool): Whether the columns of sensing matrix are normalized 
          (default True)
        axis (int): For multi-dimensional array input, the axis along which
          the linear operator will be applied 

    Returns:
        Operator: A linear operator wrapping the sensing matrix

    Examples:
        - :ref:`gallery:lop:cs_operators`
    """
    n = m if n is None else n
    Phi = crdict.rademacher_mtx(key, m, n, normalize_atoms=normalize_atoms)
    return matrix_op(Phi, axis=axis)

def sparse_binary_dict(key, m, n=None, d=None, 
    normalize_atoms=True, dense=False, axis=0):
    """An operator which represents a sparse binary sensing matrix

    Args:
        key: a PRNG key used as the random key.
        M (int): Number of rows of the sensing matrix 
        N (int): Number of columns of the sensing matrix
        d (int): Number of 1s in each column
        normalize_atoms (bool): Whether the columns of sensing matrix are normalized 
          (default True)
        dense (bool): Whether to return a dense or a sparse matrix
        axis (int): For multi-dimensional array input, the axis along which
          the linear operator will be applied 

    Returns:
        Operator: A linear operator wrapping the sensing matrix

    """
    n = m if n is None else n
    d = 10 if d is None else d
    Phi = crdict.sparse_binary_mtx(key, m, n, d=d, 
        normalize_atoms=normalize_atoms, dense=dense)
    if dense:
        return matrix_op(Phi, axis=axis)
    else:
        return sparse_real_matrix(Phi, axis=axis)


def random_orthonormal_rows_dict(key, m, n=None, axis=0):
    """An operator whose rows are orthonormal (sampled from a random orthonormal basis)

    Args:
        key: a PRNG key used as the random key.
        m (int): Number of rows of the sensing matrix 
        n (int): Number of columns of the sensing matrix
        axis (int): For multi-dimensional array input, the axis along which
          the linear operator will be applied 

    Returns:
        Operator: A linear operator wrapping the sensing matrix


    Examples:
        - :ref:`gallery:lop:cs_operators`
    """
    n = m if n is None else n
    Phi = crdict.random_orthonormal_rows(key, m, n)
    return matrix_op(Phi, axis=axis)


def random_onb_dict(key, n, axis=0):
    """An operator representing a random orthonormal basis

    Args:
        key: a PRNG key used as the random key.
        n (int): Dimension of the random orthonormal basis
        axis (int): For multi-dimensional array input, the axis along which
          the linear operator will be applied 

    Returns:
        Operator: A linear operator wrapping the sensing matrix


    Examples:
        - :ref:`gallery:lop:cs_operators`
    """
    Phi = crdict.random_onb(key, n)
    return matrix_op(Phi, axis=axis)

def binary_dict_alg(key, m, n=None, axis=0):
    """An operator representing a random matrix with 0, 1 entries.

    Args:
        key: a PRNG key used as the random key.
        m (int): Number of rows of the sensing matrix 
        n (int): Number of columns of the sensing matrix
        axis (int): For multi-dimensional array input, the axis along which
          the linear operator will be applied 

    Returns:
        Operator: A linear operator wrapping the sensing matrix

    Notes:
        It provides an algorithmic implementation of multiplying
        with a binary sensing matrix.
        This operator is far less efficient than corresponding
        matrix multiplication. Hence it is not recommended to
        be used till we figure out a more optimized implementation.
    """
    n = m if n is None else n
    keys = jax.random.split(key, n)

    def times(x):
        y = jnp.zeros(m)
        for i in range(n):
            v = jax.random.bernoulli(keys[i], shape=(m,))
            y = y + x[i] * v
        return y

    def trans(y):
        x = jnp.zeros(n)
        for i  in range(n):
            v = jax.random.bernoulli(keys[i], shape=(m,))
            x = x.at[i].set(v  @ y)
        return x

    times, trans = apply_along_axis(times, trans, axis)
    return Operator(times=times, trans=trans, shape=(n,m), real=False)
