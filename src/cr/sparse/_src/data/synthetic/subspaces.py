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

"""Random subspaces
"""

import jax.numpy as jnp
from jax import random, jit

import cr.sparse as crs

def random_subspaces(key, N, D, K):
    """Generates a set of orthonormal bases for random low dimensional subspaces

    Args:
        key: a PRNG key used as the random key.
        N (int): Dimension of the ambient space
        D (int): Dimension of the low dimensional subspace
        K (int): Number of low dimensional subspaces

    Returns:
        (:obj:`list` of :obj:`jax.numpy.ndarray`): A list of orthonormal bases for the random low dimensional subspaces
    """
    keys = random.split(key, K)
    bases = []
    for i in range(K):
        A = random.normal(keys[i], [N, D])
        Q, _ = jnp.linalg.qr(A)
        bases.append(Q)
    return bases

random_subspaces_jit = jit(random_subspaces, static_argnums=(1,2,3,))


def uniform_points_on_subspaces(key, bases, n):
    """Generates a set of nk points on the unit sphere of each of the subspaces

    Args:
        key: a PRNG key used as the random key.
        bases (:obj:`list` of :obj:`jax.numpy.ndarray`): List of orthonormal bases for the low dimensional subspaces
        n (int): Number of points on each subspace unit sphere

    Returns:
        (jax.numpy.ndarray): A matrix containing the list of points
    """
    # number of subspaces
    K = len(bases)
    # total number of points
    total = K * n
    # the ambient dimension
    N = bases[0].shape[0]
    # allocate the space
    X = jnp.zeros((N, total), dtype=bases[0].dtype)
    keys = random.split(key, K)
    start = 0
    for i in range(K):
        A = bases[i]
        # dimension of the subspace
        di = A.shape[1]
        # Generate coefficients for the subspace
        coeffs = random.normal(keys[i], [di, n])
        # Normalize the coefficients
        coeffs = crs.normalize_l2_cw(coeffs)
        # Compute the points
        points = A @ coeffs
        X = X.at[start:start+n].set(points)
        start += n
    return X
