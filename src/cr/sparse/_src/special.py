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

def _pascal_lower(n):
    A = jnp.empty((n, n), dtype=jnp.int32)
    A = A.at[0, :].set(0)
    A = A.at[:, 0].set(1)
    for i in range(1, n):
        for j in range(1, i+1):
            A = A.at[i, j].set(A[i-1, j] + A[i-1, j-1])
    return A

def _pascal_sym(n):
    A = jnp.empty((n, n), dtype=jnp.int32)
    A = A.at[0, :].set(1)
    A = A.at[:, 0].set(1)
    for i in range(1, n):
        for j in range(1, n):
            A = A.at[i, j].set(A[i-1, j] + A[i, j-1])
    return A

def pascal(n, symmetric=False):
    if symmetric:
        return _pascal_sym(n)
    else:
        return _pascal_lower(n)

pascal_jit = jax.jit(pascal, static_argnums=(0, 1))
