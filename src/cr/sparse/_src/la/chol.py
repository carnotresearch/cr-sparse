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

from cr.sparse.la import solve_Lx_b, solve_LTx_b, solve_Ux_b, solve_UTx_b


def cholesky_update_on_add_column(A, L, I, i):
    """Incrementally updates the Cholesky factorization :math:`G = L L^T` where :math:`G = A_I^T A_I with new column A_i`

    Note:

        Assumes that the first iteration is already done.
    """
    a = A[:, i]
    v = A[:, I].T @ a
    m, n = L.shape
    z = jnp.zeros((m, 1))
    w = solve_Lx_b(L, v)
    s = jnp.sqrt(a.T @ a - w.T @ w)
    L0 = jnp.hstack((L, z))
    L1 = jnp.hstack((w.T, s))
    L = jnp.vstack((L0, L1))
    return L

def cholesky_build_factor(A):
    """Builds the Cholesky factor :math:`L` of Gram matrix :math:`G = A^T A`  as :math:`G = L L^T` incrementally column wise 
    """
    a = A[:, 0]
    L = jnp.reshape(jnp.sqrt(a @ a), (1,1))
    for j in range(1, A.shape[1]):
        I = jnp.arange(0, j)
        L = cholesky_update_on_add_column(A, L, I, j)
    return L