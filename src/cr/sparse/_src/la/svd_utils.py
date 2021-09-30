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
from jax import jit, lax

from jax.scipy.linalg import svd

"""Utilities based on the Singular Value Decomposition of a matrix
"""

def orth(A, rcond=None):
    """
    Constructs an orthonormal basis for the range of A using SVD

    Args:
        A (jax.numpy.ndarray): Input matrix of size (M, N) where 
            M is the dimension of the ambient vector space and N 
            is the number of vectors in A
        rcond (float) : Relative condition number. 
            Singular values ``s`` smaller than
            ``rcond * max(s)`` are considered zero.
            Default: floating point eps * max(M,N).

    Returns:
        (jax.numpy.ndarray, int): Returns a tuple consisting of
         * the left singular vectors of A
         * the effective rank of A 

    To get the ONB, follow the two step process::

        Q, r = orth(A)
        Q = Q[:, :r]

    Examples:
        >>> A = jnp.array([[2, 0, 0], [0, 5, 0]])  # rank 2 array
        >>> Q, rank = orth(A)
        >>> print(Q)
        [[0. 1.]
        [1. 0.]]
        >>> print(rank)
        2

    The implementation is adapted from ``scipy.linalg.orth``. 
    However, the return type is different. We return the rank
    of the matrix separately. This is done so that ``orth``
    can be JIT compiled. Dynamic slices are not supported by
    JIT.
    """
    u, s, vh = svd(A, full_matrices=False)
    rank = effective_rank_from_svd(u, s, vh)
    return u, rank

orth_jit  = jit(orth, static_argnums=(1,))


def row_space(A, rcond=None):
    """
    Constructs an orthonormal basis for the row space of A using SVD

    Args:
        A (jax.numpy.ndarray): Input matrix of size (M, N) where 
            M is the dimension of the ambient vector space and N 
            is the number of vectors in A
        rcond (float) : Relative condition number. 
            Singular values ``s`` smaller than
            ``rcond * max(s)`` are considered zero.
            Default: floating point eps * max(M,N).

    Returns:
        (jax.numpy.ndarray, int): Returns a tuple consisting of
         * the right singular vectors of A
         * the effective rank of A 

    To get the ONB for the row space, follow the two step process::

        Q, r = orth(A)
        Q = Q[:, :r]

    Examples:
        >>> A = jnp.array([[2, 0, 0], [0, 5, 0]]).T
        >>> print(A)
        [[2 0]
        [0 5]
        [0 0]]
        >>> Q, rank = crla.row_space(A)
        >>> print(Q[:, :rank])
        [[0. 1.]
        [1. 0.]]
    """
    u, s, vh = svd(A, full_matrices=False)
    rank = effective_rank_from_svd(u, s, vh)
    Q = jnp.conjugate(vh.T)
    return Q, rank

row_space_jit  = jit(row_space, static_argnums=(1,))


def null_space(A, rcond=None):
    """
    Constructs an orthonormal basis for the null space of A using SVD

    Args:
        A (jax.numpy.ndarray): Input matrix of size (M, N) where 
            M is the dimension of the ambient vector space and N 
            is the number of vectors in A
        rcond (float) : Relative condition number. 
            Singular values ``s`` smaller than
            ``rcond * max(s)`` are considered zero.
            Default: floating point eps * max(M,N).

    Returns:
        (jax.numpy.ndarray, int): Returns a tuple consisting of
         * the right singular vectors of A
         * the effective rank of A 

    To get the ONB for the null space of A, follow the two step process::

        Z, r = null_space(A)
        Z = Z[:, r:]

    The dimension of the effective null space is :math:`N - r` where r is the rank of A.

    Examples:
        >>> A = random.normal(key, (3, 5))
        >>> Z, r = null_space(A)
        >>> Z = Z[:, r:]
        >>> Z.shape
        (5, 2)
        >>> print(jnp.allclose(A @ Z, 0))
        True
    """
    u, s, vh = svd(A, full_matrices=True)
    rank = effective_rank_from_svd(u, s, vh)
    N = jnp.conjugate(vh.T)
    return N, rank

null_space_jit  = jit(null_space, static_argnums=(1,))




def left_null_space(A, rcond=None):
    """
    Constructs an orthonormal basis for the left null space of A using SVD

    Args:
        A (jax.numpy.ndarray): Input matrix of size (M, N) where 
            M is the dimension of the ambient vector space and N 
            is the number of vectors in A
        rcond (float) : Relative condition number. 
            Singular values ``s`` smaller than
            ``rcond * max(s)`` are considered zero.
            Default: floating point eps * max(M,N).

    Returns:
        (jax.numpy.ndarray, int): Returns a tuple consisting of
         * the left singular vectors of A
         * the effective rank of A 

    To get the ONB for the left null space of A, follow the two step process::

        Z, r = left_null_space(A)
        Z = Z[:, r:]

    The dimension of the effective null space is :math:`M - r` where r is the rank of A.

    Examples:
        >>> A = random.normal(key, (6, 4))
        >>> Z, r = left_null_space(A)
        >>> Z = Z[:, r:]
        >>> Z.shape
        (6, 2)
        >>> print(jnp.allclose(Z.T @ A, 0))
        True
    """
    u, s, vh = svd(A, full_matrices=True)
    rank = effective_rank_from_svd(u, s, vh)
    return u, rank

left_null_space_jit  = jit(left_null_space, static_argnums=(1,))


def effective_rank(A, rcond=None):
    """
    Returns the effective rank of A based on its singular value decomposition

    Args:
        A (jax.numpy.ndarray): Input matrix of size (M, N) where 
            M is the dimension of the ambient vector space and N 
            is the number of vectors in A
        rcond (float) : Relative condition number. 
            Singular values ``s`` smaller than
            ``rcond * max(s)`` are considered zero.
            Default: floating point eps * max(M,N).

    Returns:
        (int): Returns the effective rank of A 

    Examples:
        >>> A = random.normal(key, (3, 5))
        >>> r = svd_effective_rank(A)
        >>> print(r)
        3
    """
    u, s, vh = svd(A, full_matrices=False)
    return effective_rank_from_svd(u, s, vh, rcond)

effective_rank_jit  = jit(effective_rank, static_argnums=(1,))


def effective_rank_from_svd(u, s, vh, rcond=None):
    """Returns the effective rank of a matrix from its SVD

    Args:
        u (jax.numpy.ndarray): Left singular vectors 
        s (jax.numpy.ndarray): Singular values 
        vh (jax.numpy.ndarray): Right singular vectors (Hermitian transposed)
        rcond (float) : Relative condition number. 
            Singular values ``s`` smaller than
            ``rcond * max(s)`` are considered zero.
            Default: floating point eps * max(M,N).

    Returns:
        (int): Returns the effective rank by analyzing the singular values 

    It is assumed that the SVD has already been computed.

    Examples:
        >>> A  = random.normal(key, (6, 4))
        >>> u, s, vh = jax.scipy.linalg.svd(A)
        >>> r = crla.effective_rank_from_svd(u, s, vh)
        >>> print(r)
        4
    """
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = jnp.finfo(s.dtype).eps * max(M, N)
    tol = jnp.amax(s) * rcond
    rank = jnp.sum(s > tol, dtype=int)
    return rank


@jit
def singular_values(A):
    """Returns the singular values of a matrix

    Args:
        A (jax.numpy.ndarray): Input matrix of size (M, N) where 
            M is the dimension of the ambient vector space and N 
            is the number of vectors in A


    Returns:
        (jax.numpy.ndarray): The list of singular values


    Examples:
        >>> key = random.PRNGKey(0)
        >>> A = random.normal(key, (20, 10))
        >>> print(singular_values(A))
        [6.6780386  6.19980196 5.65133988 4.89395458 4.49728071 3.9139061
         3.50887351 2.66701591 2.12520081 1.63708146]
    """
    return jnp.linalg.svd(A, full_matrices=False, compute_uv=False)
