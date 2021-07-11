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


def gaussian_dict(key, m, n, normalize_atoms=True):
    """An operator which represents a Gaussian sensix matrix (with normalized columns)
    """
    Phi = crdict.gaussian_mtx(key, m, n, normalize_atoms)
    return matrix_op(Phi)


def rademacher_dict(key, m, n):
    """An operator which represents a Rademacher sensing matrix
    """
    Phi = crdict.rademacher_mtx(key, m, n)
    return matrix_op(Phi)

def random_orthonormal_rows_dict(key, m, n):
    """An operator whose rows are orthonormal (sampled from a random orthonormal basis)
    """
    Phi = crdict.random_orthonormal_rows(key, m, n)
    return matrix_op(Phi)


def random_onb_dict(key, n):
    """An operator representing a random orthonormal basis
    """
    Phi = crdict.random_onb(key, n)
    return matrix_op(Phi)
