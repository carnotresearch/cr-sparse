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
from jax import jit
from jax import random 

def sparse_normal_representations(key, D, K, S=1):
    """
    Generates a set of sparse model vectors with normally distributed non-zero entries.
    
    * Each vector is K-sparse.
    * The non-zero basis indexes are randomly selected
      and shared among all vectors.
    * The non-zero values are normally distributed. 

    Args:
        key: a PRNG key used as the random key.
        D (int): Dimension of the model space
        K (int): Number of non-zero entries in the sparse model vectors
        S (int): Number of sparse model vectors (default 1)

    Returns:
        (jax.numpy.ndarray, jax.numpy.ndarray): A tuple consisting of 
        (i) a matrix of sparse model vectors
        (ii) an index set of locations of non-zero entries

    Example:

        >>> key = random.PRNGKey(1)
        >>> X, Omega = sparse_normal_representations(key, 6, 2, 3)
        >>> print(X.shape)
        (6, 3)
        >>> print(Omega)
        [1 5]
        >>> print(X)
        [[ 0.          0.          0.        ]
        [ 0.07545021 -1.0032069  -1.1431499 ]
        [ 0.          0.          0.        ]
        [ 0.          0.          0.        ]
        [ 0.          0.          0.        ]
        [-0.14357079  0.59042295 -1.43841705]]
    """
    r = jnp.arange(D)
    r = random.permutation(key, r)
    omega = r[:K]
    omega = jnp.sort(omega)
    shape = [K, S]
    values = random.normal(key, shape)
    result = jnp.zeros([D, S])
    result = result.at[omega, :].set(values)
    result = jnp.squeeze(result)
    return result, omega


def sparse_spikes(key, N, K, S=1):
    """
    Generates a set of sparse model vectors with Rademacher distributed non-zero entries.
    
    * Each vector is K-sparse.
    * The non-zero basis indexes are randomly selected
      and shared among all vectors.
    * Non-zero values are Rademacher distributed spikes (-1, 1). 

    Args:
        key: a PRNG key used as the random key.
        N (int): Dimension of the model space
        K (int): Number of non-zero entries in the sparse model vectors
        S (int): Number of sparse model vectors (default 1)

    Returns:
        (jax.numpy.ndarray, jax.numpy.ndarray): A tuple consisting of 
        (i) a matrix of sparse model vectors
        (ii) an index set of locations of non-zero entries

    Example:

        >>> key = random.PRNGKey(3)
        >>> X, Omega = sparse_spikes(key, 6, 2, 3)
        >>> print(X.shape)
        (6, 3)
        >>> print(Omega)
        [2 5]
        >>> print(X)
        [[ 0.  0.  0.]
        [ 0.  0.  0.]
        [-1.  1.  1.]
        [ 0.  0.  0.]
        [ 0.  0.  0.]
        [ 1. -1.  1.]]
    """
    key, subkey = random.split(key)
    perm = random.permutation(key, N)
    omega = jnp.sort(perm[:K])
    spikes = jnp.sign(random.normal(subkey, (K,S)))
    X = jnp.zeros((N, S))
    X = X.at[omega, :].set(spikes)
    # reduce the dimension if S = 1
    X = jnp.squeeze(X)
    return X, omega

