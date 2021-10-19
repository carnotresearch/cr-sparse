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

from typing import NamedTuple
import math

from jax import lax, jit, vmap
import jax.numpy as jnp
from jax.numpy.linalg import norm
from jax.scipy.linalg import svd

def bdsqr(alpha, beta, k):
    """Computes the SVD of the bidiagonal matrix
    """
    # prepare the k+1 x k bidiagonal matrix
    B = jnp.zeros((k+1, k))
    # diagonal indices for k alpha entries
    indices = jnp.diag_indices(k)
    B = B.at[indices].set(alpha[:k])
    # subdiagonal indices for k beta entries (from second row)
    rows, cols = indices
    rows = rows + 1
    B = B.at[(rows, cols)].set(beta[1:k+1])
    # print(B)
    # perform full svd
    U, s, Vh = svd(B, full_matrices=False, compute_uv=True)
    # print(s)
    # pick the last row of U as the bounds
    bnd = U[-1, :k]
    # print(bnd)
    return U, s, Vh, bnd

bdsqr_jit = jit(bdsqr, static_argnums=(2,))