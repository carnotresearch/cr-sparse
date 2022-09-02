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

"""Random subspaces
"""

import jax.numpy as jnp
from jax import random, jit

import cr.nimble as cnb

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
        key = keys[i]
        k1, k2 = random.split(key)
        A = random.normal(k1, [N, D])
        Q, R = jnp.linalg.qr(A)
        # dg = jnp.sign(jnp.diag(R))
        dg = 2 * random.bernoulli(k2, shape=(D,)) - 1
        # apply the random sign changes
        Q = cnb.diag_postmultiply(Q, dg)
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
        coeffs = cnb.normalize_l2_cw(coeffs)
        # Compute the points
        points = A @ coeffs
        X = X.at[:, start:start+n].set(points)
        start += n
    return X

uniform_points_on_subspaces_jit = jit(uniform_points_on_subspaces, static_argnums=(2,))

def two_subspaces_at_angle(key, N, D, theta):
    """Returns ONBs for two subspaces at angle theta with each other

    Args:
        key: a PRNG key used as the random key.
        N (int): Dimension of the ambient space
        D (int): Dimension of the low dimensional subspace
        theta (float): Smallest principal angle (in degrees) between the two subspaces

    Returns:
        (jax.numpy.ndarray, jax.numpy.ndarray): A tuple (A, B) consiting of two
        ONBs for the two subspaces

    Example:
        >>> rkey = random.PRNGKey(1)
        >>> N = 20
        >>> D = 4
        >>> theta = 15
        >>> A, B = two_subspaces_at_angle(rkey, N, D, theta)
        >>> print(A.shape, B.shape)
        (20, 4) (20, 4) (20, 4)
        >>> from cr.nimble.subspaces import principal_angles_deg
        >>> print(principal_angles_deg(A, B))
        [15. 90. 90. 90.]
    """
    # Convert theta to radians
    theta = jnp.deg2rad(theta)
    # Draw two random vectors in the ambient space
    X = random.normal(key, (N, 2))
    # Orthogonalize them
    U, s, VH = jnp.linalg.svd(X, full_matrices=False)
    # Pick these two vectors
    a1 = U[:, 0]
    a2 = U[:, 1]
    #  expected value of inner product between the vectors
    p = jnp.cos(theta)
    # linear combination terms for the first vector of the second space
    c1 = p
    s1 = jnp.sqrt(1 - c1**2)
    # first vector for second space
    b1 = s1 * a2 + c1 * a1
    # put these vectors together
    X = jnp.column_stack((a1, b1))
    # Find the orthogonal complement of X 
    U, s, VH = jnp.linalg.svd(X, full_matrices=True)
    Y = U[:, 2:]
    # Prepare the subspaces by picking the additional orthogonal vectors from Y
    A = jnp.column_stack((a1, Y[:, :D-1]))
    B = jnp.column_stack((b1, Y[:, D-1:2*D-2]))
    return A, B

two_subspaces_at_angle_jit = jit(two_subspaces_at_angle, static_argnums=(1,2,3))

def three_subspaces_at_angle(key, N, D, theta):
    """Returns ONBs for three subspaces at angle theta with each other

    Args:
        key: a PRNG key used as the random key.
        N (int): Dimension of the ambient space
        D (int): Dimension of the low dimensional subspace
        theta (float): Smallest principal angle (in degrees) between the three subspaces

    Returns:
        (jax.numpy.ndarray, jax.numpy.ndarray, jax.numpy.ndarray): A tuple consiting of three
        ONBs for the three subspaces

    Example:
        >>> rkey = random.PRNGKey(1)
        >>> N = 20
        >>> D = 4
        >>> theta = 15
        >>> A, B, C = three_subspaces_at_angle(rkey, N, D, theta)
        >>> print(A.shape, B.shape, C.shape)
        (20, 4) (20, 4) (20, 4)
        >>> from cr.nimble.subspaces import smallest_principal_angles_deg
        >>> angles = smallest_principal_angles_deg(jnp.array([A, B, C]))
        >>> print(jnp.round(angles, 2))
        [[ 0. 15. 15.]
        [15.  0. 15.]
        [15. 15.  0.]]
    """
    # Convert theta to radians
    theta = jnp.deg2rad(theta)
    # Draw three random vectors in the ambient space
    X = random.normal(key, (N, 3))
    # Orthogonalize them
    U, s, VH = jnp.linalg.svd(X, full_matrices=False)
    # Pick these three vectors
    a1 = U[:, 0]
    a2 = U[:, 1]
    a3 = U[:, 2]
    #  expected value of inner product between the vectors
    p = jnp.cos(theta)
    # linear combination terms for the first vector of the second space
    c1 = p
    s1 = jnp.sqrt(1 - c1**2)
    # first vector for second space
    b1 = s1 * a2 + c1 * a1
    # first vector for third space
    c1_1 = p
    c1_2 = p * (1 - p) / jnp.sqrt(1 - p**2)
    c1_3 = jnp.sqrt(1 - c1_1**2 - c1_2**2)
    c1 = c1_1 * a1 + c1_2 * a2 + c1_3 * a3
    # put these vectors together
    X = jnp.column_stack((a1, b1, c1))
    # Find the orthogonal complement of X 
    U, s, VH = jnp.linalg.svd(X, full_matrices=True)
    Y = U[:, 3:]
    # Prepare the subspaces by picking the additional orthogonal vectors from Y
    A = jnp.column_stack((a1, Y[:, :D-1]))
    B = jnp.column_stack((b1, Y[:, D-1:2*D-2]))
    C = jnp.column_stack((c1, Y[:, 2*D-2:3*D-3]))
    return A, B, C

three_subspaces_at_angle_jit = jit(three_subspaces_at_angle, static_argnums=(1,2,3))