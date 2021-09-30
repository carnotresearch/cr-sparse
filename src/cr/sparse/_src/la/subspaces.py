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

from .svd_utils import singular_values

def orth_complement(A, B):
    """Returns the orthogonal complement of A in B
    """
    rank_a = A.shape[1]
    C = jnp.hstack([A, B])
    Q, R = jnp.linalg.qr(C)
    return Q[:, rank_a:]

@jit
def principal_angles_cos(A, B):
    """Returns the cosines of principal angles between two subspaces 
    """
    AH = jnp.conjugate(A.T)
    M = AH @ B
    s = singular_values(M)
    # ensure that the singular values are below 1
    return jnp.minimum(1, s)


def principal_angles_rad(A, B):
    """Returns the principal angles between two subspaces in radians
    """
    angles = principal_angles_cos(A, B)
    return jnp.arccos(angles)


def principal_angles_deg(A, B):
    """Returns the principal angles between two subspaces in degrees
    """
    angles = principal_angles_rad(A, B)
    return jnp.rad2deg(angles)


def smallest_principal_angle_cos(A, B):
    """Returns the cosine of smallest principal angle between two subspaces 
    """
    angles = principal_angles_cos(A, B)
    return angles[0]

def smallest_principal_angle_rad(A, B):
    """Returns the smallest principal angle between two subspaces in radians
    """
    angle = smallest_principal_angle_cos(A, B)
    return jnp.arccos(angle)

def smallest_principal_angle_deg(A, B):
    """Returns the smallest principal angle between two subspaces in degrees
    """
    angle = smallest_principal_angle_rad(A, B)
    return jnp.rad2deg(angle)
