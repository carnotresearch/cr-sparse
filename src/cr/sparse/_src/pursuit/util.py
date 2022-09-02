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

from cr.nimble import solve_Lx_b, solve_LTx_b, solve_Ux_b, solve_UTx_b


def abs_max_idx(h):
    """Returns the index of entry with highest magnitude
    """
    return jnp.argmax(jnp.abs(h))

def largest_indices(h, K):
    indices = jnp.argsort(jnp.abs(h))
    return indices[:-K-1:-1]


def gram_chol_update(L, v):
    """Incrementally updates the Cholesky factorization :math:`G = L L^T` where :math:`G = \Phi^T \Phi`
    """
    m, n = L.shape
    z = jnp.zeros((m, 1))
    w = solve_Lx_b(L, v)
    s = jnp.sqrt(1  - w.T @ w)
    L0 = jnp.hstack((L, z))
    L1 = jnp.hstack((w.T, s))
    L = jnp.vstack((L0, L1))
    return L
