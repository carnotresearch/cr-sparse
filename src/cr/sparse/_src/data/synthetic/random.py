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

import math
import jax.numpy as jnp
from jax import jit, vmap
from jax import random 

import cr.nimble as cnb

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

def sparse_biuniform_representations(key, a, b, D, K, S=1):
    """
    Generates a set of sparse model vectors with bi-uniformly distributed non-zero entries.
    
    * Each vector is K-sparse.
    * The non-zero basis indexes are randomly selected
      and shared among all vectors.
    * The non-zero values have a random positive or negative sign.
    * The non-zero values have a magnitude which varies uniformly between [a,b] 

    Args:
        key: a PRNG key used as the random key.
        a (float): Minimum magnitude
        b (float): Maximum magnitude
        D (int): Dimension of the model space
        K (int): Number of non-zero entries in the sparse model vectors
        S (int): Number of sparse model vectors (default 1)

    Returns:
        (jax.numpy.ndarray, jax.numpy.ndarray): A tuple consisting of 
        (i) a matrix of sparse model vectors
        (ii) an index set of locations of non-zero entries
    """
    keys = random.split(key, 3)
    r = jnp.arange(D)
    r = random.permutation(keys[0], r)
    omega = r[:K]
    omega = jnp.sort(omega)
    shape = [K, S]
    values = random.uniform(keys[1], shape)
    values = a + (b -a) * values
    # Generate sign for non-zero entries randomly
    sgn = jnp.sign(random.normal(keys[2], shape))
    # Combine sign and magnitude
    values = sgn * values
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

def index_sets(key, N, K, S, out_axis=0):
    """Generates K-length index (sub)sets of the set [0..N-1] for S signals
    """
    omega = jnp.arange(N)
    keys = random.split(key, S)
    set_gen = lambda key : random.permutation(key, omega)[:K] 
    I = vmap(set_gen, 0, out_axis)(keys)
    return I


def points_orthogonal_to(key, x, S):
    """Generates a  set of points which are orthogonal to a given point x
    """
    # first we normalize x
    norm_x = cnb.arr_l2norm(x)
    x = x / norm_x
    # shape for the output array
    shape = (S,) + x.shape
    points = random.normal(key, shape)
    correlations = points  @ x
    projections = correlations[:, jnp.newaxis]  * x[jnp.newaxis, :]
    return points - projections


########################################################
#
# Group/Block Sparsity
#
########################################################

def sparse_normal_blocks(key, D: int, K: int, B: int, S: int=1, 
    cor: float =0.,
    normalize_blocks=False):
    """Generates representations where some blocks have normally distributed
    coefficients while others are zero.


    Args:
        key: a PRNG key used as the random key.
        D (int): Dimension of the model space
        K (int): Number of nonzero blocks
        B (int): Length of each block
        S (int): Number of sparse model vectors (default 1)
        cor (float): Intra block correlation under AR-1 model
        normalize_blocks (bool): Normalize the nonzero coefficients in each block

    Returns:
        A tuple consisting of
        (i) a vector/matrix of sparse model vectors
        (ii) active block numbers
        (iii) indices corresponding to the locations of nonzero entries

    Notes:
    - Each active block of samples is normalized to unit norm.
    """
    # Compute the number of blocks
    C = D // B
    # Make sure that there is nothing left
    if D != C * B:
        raise ValueError(f"{D} must be a multiple of {B}.")
    key = random.split(key)
    # Identify the blocks which will be activated
    blocks = random.choice(key[0], C, shape=(K,), replace=False)
    blocks = jnp.sort(blocks)
    indices = jnp.arange(B)
    #+ jnp.broadcast_to(blocks, indices.shape)
    indices = indices[None, :] + (B * blocks)[:, None]
    indices = indices.flatten()
    # number of nonzero coefficients for each vector
    nv = K*B 
    # total number of coefficients
    n = nv*S
    # total number of blocks
    nb = K * S
    vshape = (K*B,S)
    # random values
    values = random.normal(key[1], (nb,B))
    if cor:
        # We need to enforce correlation
        # among the coefficients within a block
        b1 = cor
        b2 = math.sqrt(1 - b1 **2)
        for i in range(1, B):
            cur = values[:, i]
            prev = values[:, i-1]
            new = b1 * prev + b2 * cur
            values = values.at[:, i].set(new)
    if normalize_blocks:
        values = cnb.normalize_l2_rw(values)
    values = jnp.reshape(values, (S, nv)).T

    x = jnp.zeros([D,S])
    # print(x, len(x), len(indices), len(values))
    x = x.at[indices, :].set(values)
    x = jnp.squeeze(x)
    return x, blocks, indices

