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

from cr.sparse import hermitian

def gram(A):
    """Computes the Gram matrix :math:`G = A^T A`
    """
    if jnp.isrealobj(A):
        return A.T @ A
    G = hermitian(A) @ A
    return G.real

def frame(A):
    """Computes the frame matrix :math:`G = A A^T`
    """
    if jnp.isrealobj(A):
        return A @ A.T
    F = A @ hermitian(A)
    return F.real


def coherence_with_index(A):
     G = gram(A)
     G = jnp.abs(G)
     n = G.shape[0]
     # set diagonals to 0
     G = G.at[jnp.diag_indices(n)].set(0)
     index = jnp.unravel_index(jnp.argmax(G, axis=None), G.shape)
     max_val = G[index]
     return max_val, index

def coherence(A):
    """Computes the coherence of a dictionary
    """
    max_val, index = coherence_with_index(A)
    return max_val


def frame_bounds(A):
    """Computes the frame bounds (largest and smallest singular valuee)
    """
    s = jnp.linalg.svd(A, False, False)
    indices = jnp.array([0, -1])
    return s[indices]

def upper_frame_bound(A):
    """Computes the upper frame bound for a dictionary
    """
    s = jnp.linalg.svd(A, False, False)
    return s[0]

def lower_frame_bound(A):
    """Computes the lower frame bound for a dictionary
    """
    s = jnp.linalg.svd(A, False, False)
    return s[-1]


def babel(A):
    """Computes the babel function for a dictionary (generalized coherence)
    """
    # compute gram matrix
    G = gram(A)
    # compute absolute values
    G = jnp.abs(G)
    # sort on each row
    G = jnp.sort(G)
    # reverse each row and drop last entry [self similarity is 1]
    G = G[:, -2::-1]
    # compute cumulative sums over rows
    sums = jnp.cumsum(G, axis=1)
    # find maximum over each column
    result = jnp.max(sums, axis=0)
    return result