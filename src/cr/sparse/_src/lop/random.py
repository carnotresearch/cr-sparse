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
from .basic import matrix as matrix_op
import cr.sparse.dict as crdict


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
