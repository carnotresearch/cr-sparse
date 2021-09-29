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
from jax import random

def random_subspaces(key, N, D, K):
    """Generates a set of orthonormal bases for random low dimensional subspaces

    Args:
        key: a PRNG key used as the random key.
        N (int): Dimension of the ambient space
        D (int): Dimension of the low dimensional subspace
        K (int): Number of low dimensional subspaces

    Returns:
        (list): A list of orthonormal bases for the random low dimensional subspaces
    """
    keys = random.split(key, K)
    bases = []
    for i in range(K):
        A = random.normal(key, [N, D])
        Q, _ = jnp.linalg.qr(A)
        bases.append(Q)
    return bases
