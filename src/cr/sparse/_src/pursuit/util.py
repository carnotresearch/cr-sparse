# Copyright 2021 Carnot Research Pvt Ltd
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

from cr.sparse.la import solve_Lx_b, solve_LTx_b, solve_Ux_b, solve_UTx_b


def abs_max_idx(h):
    """Returns the index of entry with highest magnitude
    """
    return jnp.argmax(jnp.abs(h))

def gram_chol_update(L, v):
    m, n = L.shape
    z = jnp.zeros((m, 1))
    w = solve_Lx_b(L, v)
    s = jnp.sqrt(1  - w.T @ w)
    L0 = jnp.hstack((L, z))
    L1 = jnp.hstack((w.T, s))
    L = jnp.vstack((L0, L1))
    return L