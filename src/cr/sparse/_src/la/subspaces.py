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

from jax.numpy.linalg import norm

from .svd_utils import singular_values
from .util import hermitian, AH_v

def orth_complement(A, B):
    """Returns the orthogonal complement of A in B
    """
    rank_a = A.shape[1]
    C = jnp.hstack([A, B])
    Q, R = jnp.linalg.qr(C)
    return Q[:, rank_a:]

def principal_angles_cos(A, B):
    """Returns the cosines of principal angles between two subspaces 

    Args:
        A (jax.numpy.ndarray): ONB for the first subspace
        B (jax.numpy.ndarray): ONB for the second subspace

    Returns:
        (jax.numpy.ndarray): The list of principal angles between two subspaces 
            from smallest to the largest.
    """
    AH = jnp.conjugate(A.T)
    M = AH @ B
    s = singular_values(M)
    # ensure that the singular values are below 1
    return jnp.minimum(1, s)

principal_angles_cos_jit = jit(principal_angles_cos)

def principal_angles_rad(A, B):
    """Returns the principal angles between two subspaces in radians

    Args:
        A (jax.numpy.ndarray): ONB for the first subspace
        B (jax.numpy.ndarray): ONB for the second subspace

    Returns:
        (jax.numpy.ndarray): The list of principal angles between two subspaces 
            from smallest to the largest.
    """
    angles = principal_angles_cos(A, B)
    return jnp.arccos(angles)

principal_angles_rad_jit = jit(principal_angles_rad)

def principal_angles_deg(A, B):
    """Returns the principal angles between two subspaces in degrees

    Args:
        A (jax.numpy.ndarray): ONB for the first subspace
        B (jax.numpy.ndarray): ONB for the second subspace

    Returns:
        (jax.numpy.ndarray): The list of principal angles between two subspaces 
            from smallest to the largest.
    """
    angles = principal_angles_rad(A, B)
    return jnp.rad2deg(angles)

principal_angles_deg_jit = jit(principal_angles_deg)

def smallest_principal_angle_cos(A, B):
    """Returns the cosine of smallest principal angle between two subspaces 

    Args:
        A (jax.numpy.ndarray): ONB for the first subspace
        B (jax.numpy.ndarray): ONB for the second subspace

    Returns:
        (float): Cosine of the smallest principal angle between the two subspaces
    """
    angles = principal_angles_cos(A, B)
    return angles[0]

smallest_principal_angle_cos_jit = jit(smallest_principal_angle_cos)

def smallest_principal_angle_rad(A, B):
    """Returns the smallest principal angle between two subspaces in radians

    Args:
        A (jax.numpy.ndarray): ONB for the first subspace
        B (jax.numpy.ndarray): ONB for the second subspace

    Returns:
        (float): The smallest principal angle between the two subspaces in radians
    """
    angle = smallest_principal_angle_cos(A, B)
    return jnp.arccos(angle)

smallest_principal_angle_rad_jit = jit(smallest_principal_angle_rad)

def smallest_principal_angle_deg(A, B):
    """Returns the smallest principal angle between two subspaces in degrees

    Args:
        A (jax.numpy.ndarray): ONB for the first subspace
        B (jax.numpy.ndarray): ONB for the second subspace

    Returns:
        (float): The smallest principal angle between the two subspaces in degrees
    """
    angle = smallest_principal_angle_rad(A, B)
    return jnp.rad2deg(angle)

smallest_principal_angle_deg_jit = jit(smallest_principal_angle_deg)

def smallest_principal_angles_cos(subspaces):
    """Returns the smallest principal angles between each pair of subspaces

    Args:
        A (jax.numpy.ndarray): An array of ONBs for the subspaces

    Returns:
        (jax.numpy.ndarray): A symmetric matrix containing the cosine of the 
            smallest principal angles between each pair of subspaces

    Further reading on implementation:

    * `Vectorizing computations on pairs of elements in an nd-array <https://towardsdatascience.com/vectorizing-computations-on-pairs-of-elements-in-an-nd-array-326b5a648ad6>`_
    * `SO: How to vectorize a 2 level loop in NumPy <https://stackoverflow.com/questions/69391894/how-to-vectorize-a-2-level-loop-in-numpy>`_
    """
    subspaces = jnp.asarray(subspaces)
    # Number of subspaces
    k = subspaces.shape[0]
    # Indices for upper triangular matrix
    i, j = jnp.triu_indices(k, k=1)
    # prepare all the possible pairs of A and B
    A = subspaces[i]
    B = subspaces[j]
    AH = jnp.conjugate(jnp.transpose(A, axes=(0,2,1)))
    M = jnp.matmul(AH, B)
    s = jnp.linalg.svd(M, compute_uv=False)
    # keep only the first index
    s = s[:, 0]
    # prepare the returning matrix
    r = jnp.eye(k)
    r = r.at[i, j].set(s)
    r = r + r.T - jnp.eye(k)
    # make sure that there is no overflow
    r = jnp.minimum(r, 1.)
    return r


smallest_principal_angles_cos_jit = jit(smallest_principal_angles_cos)


def smallest_principal_angles_rad(subspaces):
    """Returns the smallest principal angles between each pair of subspaces in radians

    Args:
        A (jax.numpy.ndarray): An array of ONBs for the subspaces

    Returns:
        (jax.numpy.ndarray): A symmetric matrix containing the 
            smallest principal angles between each pair of subspaces in radians
    """
    result = smallest_principal_angles_cos(subspaces)
    return jnp.arccos(result)

smallest_principal_angles_rad_jit = jit(smallest_principal_angles_rad)

def smallest_principal_angles_deg(subspaces):
    """Returns the smallest principal angles between each pair of subspaces in degrees

    Args:
        A (jax.numpy.ndarray): An array of ONBs for the subspaces

    Returns:
        (jax.numpy.ndarray): A symmetric matrix containing the 
            smallest principal angles between each pair of subspaces in degrees
    """
    result = smallest_principal_angles_rad(subspaces)
    return jnp.rad2deg(result)

smallest_principal_angles_deg_jit = jit(smallest_principal_angles_deg)


def subspace_distance(A, B):
    r"""Returns the Grassmannian distance between two subspaces

    Args:
        A (jax.numpy.ndarray): ONB for the first subspace
        B (jax.numpy.ndarray): ONB for the second subspace

    Returns:
        (float): Distance between the two subspaces

    the `Grassmannian <https://en.wikipedia.org/wiki/Grassmannian>`_ 
    is a space that parameterizes  all  k dimensional linear 
    subspaces of a vector space V. 
    A `metric <https://math.stackexchange.com/questions/198111/distance-between-real-finite-dimensional-linear-subspaces>`_
    can be defined over this space. We can use this metric
    to compute the distance between two subspaces.
    """
    # Compute the projection operators for the two subspaces
    PA = A @ jnp.conjugate(A.T)
    PB = B @ jnp.conjugate(B.T)
    # Difference between the projection operators
    D = PA  - PB
    # Return the operator norm of D
    return jnp.linalg.norm(D)

subspace_distance_jit = jit(subspace_distance)


def project_to_subspace(U, v):
    """Projects a vector to a subspace

    Args:
        U (jax.numpy.ndarray): ONB for the subspace
        v (jax.numpy.ndarray): A vector in the ambient space

    Returns:
        (jax.numpy.ndarray): Projection of v onto the subspace spanned by U

    Example:
        >>> A = jnp.eye(6)[:, :3]
        >>> v = jnp.arange(6) + 0.
        >>> u = project_to_subspace(A, v)
        >>> print(A)
        [[1. 0. 0.]
        [0. 1. 0.]
        [0. 0. 1.]
        [0. 0. 0.]
        [0. 0. 0.]
        [0. 0. 0.]]
        >>> print(v)
        [0. 1. 2. 3. 4. 5.]
        >>> print(u)
        [0. 1. 2. 0. 0. 0.]
    """
    UHv = AH_v(U, v)
    return U  @ UHv

def is_in_subspace(U, v):
    """Checks whether a vector v is in the subspace spanned by an ONB U or not

    Args:
        U (jax.numpy.ndarray): ONB for the subspace
        v (jax.numpy.ndarray): A vector in the ambient space

    Returns:
        (bool): True if v lies in the subspace spanned by U, False otherwise

    Example:
        >>> A = jnp.eye(6)[:, :3]
        >>> v = jnp.arange(6) + 0.
        >>> print(is_in_subspace(A, v))
        False
        >>> u = project_to_subspace(A, v)
        >>> print(is_in_subspace(A, u))
        True
    """
    # Compute the projection
    p = project_to_subspace(U, v)
    # Compute the error
    e = p - v
    nv = norm(v)
    ne = norm(e)
    return ne <= 1e-6 * nv