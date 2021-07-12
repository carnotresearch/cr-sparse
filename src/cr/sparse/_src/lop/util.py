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

import jax
import jax.numpy as jnp


def to_matrix(A):
    """Converts a linear operator to a matrix"""
    n = A.shape[1]
    if not A.linear:
        raise Exception("This operator is not linear.")
    I = jnp.eye(n)
    return jax.vmap(A.times, (1), (1))(I)

def to_adjoint_matrix(A):
    """Converts the adjoint of a linear operator to a matrix"""
    if not A.linear:
        raise Exception("This operator is not linear.")
    m = A.shape[0]
    I = jnp.eye(m)
    return jax.vmap(A.trans, (1), (1))(I)


def to_complex_matrix(A):
    """Converts a linear operator to a matrix in complex numbers"""
    if not A.linear:
        raise Exception("This operator is not linear.")
    n = A.shape[1]
    I = jnp.eye(n) + 0j
    return jax.vmap(A.times, (1), (1))(I)
