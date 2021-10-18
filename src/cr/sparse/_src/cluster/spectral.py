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

"""
Spectral Clustering
"""

from typing import NamedTuple

from jax import lax, jit, vmap, random
import jax.numpy as jnp
from jax.numpy.linalg import norm

from .kmeans import kmeans

import cr.sparse as crs

class SpectralclusteringSolution(NamedTuple):
    """The solution for K-means algorithm
    """
    laplancian : jnp.ndarray
    """The Laplacian"""
    singular_values: jnp.ndarray
    """Singular values of the Laplancian"""
    assignment: jnp.ndarray
    """Current assignment of points to centroids"""
    num_clusters: int
    """The number of clusters"""
    connectivity: float
    """Graph connectivity"""


def unnormalized_laplacian(W):
    # Compute the degree
    D = jnp.diag(jnp.sum(W, 0))
    # Compute the Laplacian
    L = D - W
    return L


def unnormalized(key, W):
    """Unnormalized spectral clustering

    Args:
        key: a PRNG key used for the k-means algorithm
        W (jax.numpy.ndarray): Similarity/Weights matrix

    Returns:
        (SpectralclusteringSolution): A named tuple with the spectral clustering 
        solution (Laplacian, singular values, cluster assignment)
    """
    # make sure that W is square
    m, n = W.shape
    assert m == n, "W must be square"
    # Compute the Laplacian
    L = unnormalized_laplacian(W)
    # Compute the SVD of the Laplacian
    U, S, VH = jnp.linalg.svd(L)
    #print(jnp.round(S, 2))
    # we need to look from the smaller singular value side
    # smallest one will be 0.
    sdiff = jnp.diff(S[:-1])
    #print(sdiff)
    index = jnp.argmin(sdiff)
    #print(index)
    # number of clusters
    k = n - index - 1
    # Choose the last k eigen vectors
    # TODO this step cannot be JITTED
    kernel = VH.T[:,n-k:]
    # TODO we cannot use JITTED kmeans since k itself is dynamic
    result = kmeans(key, kernel, k, iter=100)
    return SpectralclusteringSolution(singular_values=S, 
        assignment=result.assignment,
        laplancian=L,
        num_clusters=k,
        connectivity=S[-2])

def unnormalized_k(key, W, k):
    """Unnormalized spectral clustering with known number of clusters

    Args:
        key: a PRNG key used for the k-means algorithm
        W (jax.numpy.ndarray): Similarity/Weights matrix
        k (int): The number of clusters

    Returns:
        (SpectralclusteringSolution): A named tuple with the spectral clustering 
        solution (Laplacian, singular values, cluster assignment)
    """
    # make sure that W is square
    m, n = W.shape
    assert m == n, "W must be square"
    # Compute the Laplacian
    L = unnormalized_laplacian(W)
    # Compute the SVD of the Laplacian
    U, S, VH = jnp.linalg.svd(L)
    # Choose the last k eigen vectors
    kernel = VH.T[:,n-k:]
    result = kmeans(key, kernel, k, iter=100)
    return SpectralclusteringSolution(singular_values=S, 
        assignment=result.assignment,
        laplancian=L,
        num_clusters=k,
        connectivity=S[-2])

unnormalized_k_jit = jit(unnormalized_k, static_argnums=(2,))

def normalized_random_walk_laplacian(W):
    # Compute the degree
    D = jnp.sum(W, 0)
    D_inv = D**(-1)
    # Compute the Laplacian
    # L = I - D_inv @ W
    L = crs.add_to_diagonal(-crs.diag_premultiply(D_inv, W), 1.)
    return L

normalized_random_walk_laplacian_jit = jit(normalized_random_walk_laplacian)

def normalized_random_walk(key, W):
    """Normalized spectral clustering with random walk

    Args:
        key: a PRNG key used for the k-means algorithm
        W (jax.numpy.ndarray): Similarity/Weights matrix

    Returns:
        (SpectralclusteringSolution): A named tuple with the spectral clustering 
        solution (Laplacian, singular values, cluster assignment)
    """
    # make sure that W is square
    m, n = W.shape
    assert m == n, "W must be square"
    # Compute the Laplacian
    L = normalized_random_walk_laplacian(W)
    # Compute the SVD of the Laplacian
    U, S, VH = jnp.linalg.svd(L)
    # we need to look from the smaller singular value side
    # smallest one will be 0.
    sdiff = jnp.diff(S[:-1])
    index = jnp.argmin(sdiff)
    # number of clusters
    k = n - index - 1
    # Choose the last k eigen vectors
    # TODO this step cannot be JITTED
    kernel = VH.T[:,n-k:]
    # TODO we cannot use JITTED kmeans since k itself is dynamic
    result = kmeans(key, kernel, k, iter=100)
    return SpectralclusteringSolution(singular_values=S, 
        assignment=result.assignment,
        laplancian=L,
        num_clusters=k,
        connectivity=S[-2])


def normalized_random_walk_k(key, W, k):
    """Normalized spectral clustering with random walk

    Args:
        key: a PRNG key used for the k-means algorithm
        W (jax.numpy.ndarray): Similarity/Weights matrix
        k (int): The number of clusters

    Returns:
        (SpectralclusteringSolution): A named tuple with the spectral clustering 
        solution (Laplacian, singular values, cluster assignment)
    """
    # make sure that W is square
    m, n = W.shape
    assert m == n, "W must be square"
    # Compute the Laplacian
    L = normalized_random_walk_laplacian(W)
    # Compute the SVD of the Laplacian
    U, S, VH = jnp.linalg.svd(L)
    # Choose the last k eigen vectors
    kernel = VH.T[:,n-k:]
    result = kmeans(key, kernel, k, iter=100)
    return SpectralclusteringSolution(singular_values=S, 
        assignment=result.assignment,
        laplancian=L,
        num_clusters=k,
        connectivity=S[-2])

normalized_random_walk_k_jit = jit(normalized_random_walk_k, static_argnums=(2,))