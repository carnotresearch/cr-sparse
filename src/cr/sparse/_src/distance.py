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

"""Pairwise distances between a set of points
"""
from jax import jit
import jax.numpy as jnp

@jit
def pairwise_sqr_l2_distances_rw(A, B):
    r"""Computes the pairwise squared distances between points in A and points in B where each point is a row vector

    Args:
        A (jax.numpy.ndarray): A set of M K-dimensional points (row-wise)
        B (jax.numpy.ndarray): A set of N K-dimensional points (row-wise)

    Returns:
        (jax.numpy.ndarray): An MxN matrix D of squared distances 
        between points in A and points in B

    * Let the ambient space of points be :math:`\mathbb{F}^K`.
    * :math:`A` contains the points :math:`a_i` with :math:`1 \leq i \leq M` 
      and each point maps to a row of :math:`A`.
    * :math:`B` contains the points :math:`b_j` with :math:`1 \leq j \leq N` 
      and each point maps to a row of :math:`B`.

    Then the distance matrix :math:`D` is of size :math:`M \times N` and consists of:

    .. math::

        d_{i, j} = \| a_i - b_j \|_2^2 = \langle a_i - b_j , a_i - b_j  \rangle
    """
    M = A.shape[0]
    N = B.shape[0]
    # compute squared sums for each row vector
    a_sums = jnp.sum(A*A, axis=1)
    # reshape to Mx1 column vector
    a_sums = jnp.reshape(a_sums, (M, 1))
    # broadcast to MxN matrix
    a_sums = a_sums * jnp.ones((1,N))

    # compute squared sums for each row vector
    b_sums = jnp.sum(B*B, axis=1)
    # broadcast to MxN matrix
    b_sums = b_sums * jnp.ones((M, 1))

    # multiply A (M x p) and B.T (p x N)
    prods = A @ B.T 
    return a_sums + b_sums - 2 * prods

@jit
def pairwise_sqr_l2_distances_cw(A, B):
    r"""Computes the pairwise squared distances between points in A and points in B where each point is a column vector

    Args:
        A (jax.numpy.ndarray): A set of M K-dimensional points (column-wise)
        B (jax.numpy.ndarray): A set of N K-dimensional points (column-wise)

    Returns:
        (jax.numpy.ndarray): An MxN matrix D of squared distances 
        between points in A and points in B

    * Let the ambient space of points be :math:`\mathbb{F}^K`.
    * :math:`A` contains the points :math:`a_i` with :math:`1 \leq i \leq M` 
      and each point maps to a column of :math:`A`.
    * :math:`B` contains the points :math:`b_j` with :math:`1 \leq j \leq N` 
      and each point maps to a column of :math:`B`.

    Then the distance matrix :math:`D` is of size :math:`M \times N` and consists of:

    .. math::

        d_{i, j} = \| a_i - b_j \|_2^2 = \langle a_i - b_j , a_i - b_j  \rangle

    """
    M = A.shape[1]
    N = B.shape[1]
    # compute squared sums for each column vector
    a_sums = jnp.sum(A*A, axis=0)
    # reshape to Mx1 column vector
    a_sums = jnp.reshape(a_sums, (M, 1))
    # broadcast to MxN matrix
    a_sums = a_sums * jnp.ones((1,N))

    # compute squared sums for each column vector
    b_sums = jnp.sum(B*B, axis=0)
    # broadcast to MxN matrix
    b_sums = b_sums * jnp.ones((M, 1))

    # multiply A.T (M x p) and B (p x N)
    prods = A.T @ B 
    return a_sums + b_sums - 2 * prods


@jit
def pairwise_l2_distances_rw(A, B):
    r"""Computes the pairwise distances between points in A and points in B where each point is a row vector

    Args:
        A (jax.numpy.ndarray): A set of M K-dimensional points (row-wise)
        B (jax.numpy.ndarray): A set of N K-dimensional points (row-wise)

    Returns:
        (jax.numpy.ndarray): An MxN matrix D of euclidean distances 
        between points in A and points in B
    """
    return jnp.sqrt(pairwise_sqr_l2_distances_rw(A, B))


@jit
def pairwise_l2_distances_cw(A, B):
    r"""Computes the pairwise distances between points in A and points in B where each point is a column vector

    Args:
        A (jax.numpy.ndarray): A set of M K-dimensional points (column-wise)
        B (jax.numpy.ndarray): A set of N K-dimensional points (column-wise)

    Returns:
        (jax.numpy.ndarray): An MxN matrix D of euclidean distances 
        between points in A and points in B
    """
    return jnp.sqrt(pairwise_sqr_l2_distances_cw(A, B))

@jit
def pdist_sqr_l2_rw(A):
    r"""Computes the pairwise squared distances between points in A where each point is a row vector

    Args:
        A (jax.numpy.ndarray): A set of N K-dimensional points (row-wise)

    Returns:
        (jax.numpy.ndarray): An NxN matrix D of squared euclidean distances 
        between points in A

    * Let the ambient space of points be :math:`\mathbb{F}^K`.
    * :math:`A` contains the points :math:`a_i` with :math:`1 \leq i \leq N` 
      and each point maps to a row of :math:`A`.

    Then the distance matrix :math:`D` is of size :math:`N \times N` and consists of:

    .. math::

        d_{i, j} = \| a_i - a_j \|_2^2 = \langle a_i - a_j , a_i - a_j  \rangle
    """
    M = A.shape[0]
    # compute squared sums for each row vector
    sums = jnp.sum(A*A, axis=1)
    # broadcast to MxM matrix
    a_sums = jnp.reshape(sums, (M,1)) * jnp.ones((1, M))
    b_sums = sums * jnp.ones((M, 1))

    # multiply A (M x p) and A.T (p x M)
    prods = A @ A.T 
    return a_sums + b_sums - 2*prods

@jit
def pdist_sqr_l2_cw(A):
    r"""Computes the pairwise squared distances between points in A where each point is a column vector

    Args:
        A (jax.numpy.ndarray): A set of N K-dimensional points (column-wise)

    Returns:
        (jax.numpy.ndarray): An NxN matrix D of squared euclidean distances 
        between points in A

    * Let the ambient space of points be :math:`\mathbb{F}^K`.
    * :math:`A` contains the points :math:`a_i` with :math:`1 \leq i \leq N` 
      and each point maps to a column of :math:`A`.

    Then the distance matrix :math:`D` is of size :math:`N \times N` and consists of:

    .. math::

        d_{i, j} = \| a_i - a_j \|_2^2 = \langle a_i - a_j , a_i - a_j  \rangle
    """
    M = A.shape[1]
    # compute squared sums for each col vector
    sums = jnp.sum(A*A, axis=0)
    # broadcast to MxN matrix
    a_sums = jnp.reshape(sums, (M, 1)) * jnp.ones((1,M))
    b_sums = sums * jnp.ones((M, 1))
    # multiply A.T (M x p) and A (p x M)
    prods = A.T @ A 
    return a_sums + b_sums - 2 * prods

@jit
def pdist_l2_rw(A):
    r"""Computes the pairwise distances between points in A where ach point is a row vector

    Args:
        A (jax.numpy.ndarray): A set of N K-dimensional points (row-wise)

    Returns:
        (jax.numpy.ndarray): An NxN matrix D of euclidean distances 
        between points in A
    """
    return jnp.sqrt(pdist_sqr_l2_rw(A))

@jit
def pdist_l2_cw(A):
    r"""Computes the pairwise distances between points in A where each point is a column vector

    Args:
        A (jax.numpy.ndarray): A set of N K-dimensional points (column-wise)

    Returns:
        (jax.numpy.ndarray): An NxN matrix D of euclidean distances 
        between points in A
    """
    return jnp.sqrt(pdist_sqr_l2_cw(A))


@jit
def pairwise_l1_distances_rw(A, B):
    r"""Computes the pairwise city-block distances between points in A and points in B where each point is a row vector

    Args:
        A (jax.numpy.ndarray): A set of M K-dimensional points (row-wise)
        B (jax.numpy.ndarray): A set of N K-dimensional points (row-wise)

    Returns:
        (jax.numpy.ndarray): An MxN matrix D of city-block distances 
        between points in A and points in B
    """
    return jnp.sum(jnp.abs(A[:, None, :] - B[None, :, :]), axis=-1)

@jit
def pairwise_l1_distances_cw(A, B):
    r"""Computes the pairwise city-block distances between points in A and points in B where each point is a column vector

    Args:
        A (jax.numpy.ndarray): A set of M K-dimensional points (column-wise)
        B (jax.numpy.ndarray): A set of N K-dimensional points (column-wise)

    Returns:
        (jax.numpy.ndarray): An MxN matrix D of city-block distances 
        between points in A and points in B
    """
    return jnp.sum(jnp.abs(A[:, :, None] - B[:, None, :]), axis=0)

@jit
def pdist_l1_rw(A):
    r"""Computes the pairwise city-block distances between points in A where each point is a row vector

    Args:
        A (jax.numpy.ndarray): A set of M K-dimensional points (row-wise)

    Returns:
        (jax.numpy.ndarray): An MxM matrix D of city-block distances 
        between points in A
    """
    return pairwise_l1_distances_rw(A, A)

@jit
def pdist_l1_cw(A):
    r"""Computes the pairwise city-block distances between points in A where each point is a column vector

    Args:
        A (jax.numpy.ndarray): A set of M K-dimensional points (column-wise)

    Returns:
        (jax.numpy.ndarray): An MxM matrix D of city-block distances 
        between points in A
    """
    return pairwise_l1_distances_cw(A, A)


@jit
def pairwise_linf_distances_rw(A, B):
    r"""Computes the pairwise Chebyshev distances between points in A and points in B where each point is a row vector

    Args:
        A (jax.numpy.ndarray): A set of M K-dimensional points (row-wise)
        B (jax.numpy.ndarray): A set of N K-dimensional points (row-wise)

    Returns:
        (jax.numpy.ndarray): An MxN matrix D of Chebyshev distances 
        between points in A and points in B
    """
    return jnp.max(jnp.abs(A[:, None, :] - B[None, :, :]), axis=-1)

@jit
def pairwise_linf_distances_cw(A, B):
    r"""Computes the pairwise Chebyshev distances between points in A and points in B where each point is a column vector

    Args:
        A (jax.numpy.ndarray): A set of M K-dimensional points (column-wise)
        B (jax.numpy.ndarray): A set of N K-dimensional points (column-wise)

    Returns:
        (jax.numpy.ndarray): An MxN matrix D of Chebyshev distances 
        between points in A and points in B
    """
    return jnp.max(jnp.abs(A[:, :, None] - B[:, None, :]), axis=0)


@jit
def pdist_linf_rw(A):
    r"""Computes the pairwise Chebyshev distances between points in A where each point is a row vector

    Args:
        A (jax.numpy.ndarray): A set of M K-dimensional points (row-wise)

    Returns:
        (jax.numpy.ndarray): An MxM matrix D of Chebyshev distances 
        between points in A
    """
    return pairwise_linf_distances_rw(A, A)

@jit
def pdist_linf_cw(A):
    r"""Computes the pairwise Chebyshev distances between points in A where each point is a column vector

    Args:
        A (jax.numpy.ndarray): A set of M K-dimensional points (column-wise)

    Returns:
        (jax.numpy.ndarray): An MxM matrix D of Chebyshev distances 
        between points in A
    """
    return pairwise_linf_distances_cw(A, A)

