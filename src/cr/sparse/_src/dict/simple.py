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

import math
from functools import partial

import numpy as np
import scipy

import jax.numpy as jnp
from jax import random
from jax import jit


from cr.sparse import normalize_l2_cw, promote_arg_dtypes, hermitian


def gaussian_mtx(key, N, D, normalize_atoms=True):
    """A dictionary/sensing matrix where entries are drawn independently from normal distribution.

    Args:
        key: a PRNG key used as the random key.
        N (int): Number of rows of the sensing matrix 
        D (int): Number of columns of the sensing matrix
        normalize_atoms (bool): Whether the columns of sensing matrix are normalized 
          (default True)

    Returns:
        (jax.numpy.ndarray): A Gaussian sensing matrix of shape (N, D)

    Example:

        >>> from jax import random
        >>> import cr.sparse as crs
        >>> import cr.sparse.dict
        >>> m, n = 8, 16
        >>> Phi = cr.sparse.dict.gaussian_mtx(random.PRNGKey(0), m, n)
        >>> print(Phi.shape)
        (8, 16)
        >>> print(crs.norms_l2_cw(Phi))
        [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        >>> print(cr.sparse.dict.coherence(Phi))
        [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        >>> print(cr.sparse.dict.babel(Phi))
        [0.85866616 1.59791754 2.13943785 2.61184779 2.9912899  3.38281051
        3.74641682 4.08225813 4.29701559 4.49942648 4.68680188 4.83106192
        4.95656728 5.05541184 5.10697535]
    """
    shape = (N, D)
    dict = random.normal(key, shape)
    if normalize_atoms:
        dict = normalize_l2_cw(dict)
    else:
        sigma = math.sqrt(N)
        dict = dict / sigma
    return dict

def rademacher_mtx(key, M, N, normalize_atoms=True):
    """A dictionary/sensing matrix where entries are drawn independently from Rademacher distribution.

    Args:
        key: a PRNG key used as the random key.
        M (int): Number of rows of the sensing matrix 
        N (int): Number of columns of the sensing matrix
        normalize_atoms (bool): Whether the columns of sensing matrix are normalized 
          (default True)

    Returns:
        (jax.numpy.ndarray): A Rademacher sensing matrix of shape (M, N)

    Example:

        >>> from jax import random
        >>> import cr.sparse as crs
        >>> import cr.sparse.dict
        >>> m, n = 8, 16
        >>> Phi = cr.sparse.dict.rademacher_mtx(random.PRNGKey(0), m, n, 
              normalize_atoms=False)
        >>> print(Phi)
        [[ 1.  1.  1. -1. -1.  1. -1. -1.  1.  1.  1. -1.  1.  1. -1.  1.]
        [-1.  1.  1.  1. -1.  1.  1.  1. -1. -1. -1.  1. -1. -1. -1.  1.]
        [ 1. -1. -1.  1. -1.  1.  1. -1.  1. -1.  1. -1. -1. -1.  1. -1.]
        [ 1. -1. -1.  1. -1.  1.  1. -1.  1. -1. -1.  1.  1.  1. -1.  1.]
        [-1. -1. -1. -1. -1. -1. -1.  1. -1. -1. -1.  1. -1. -1. -1. -1.]
        [-1.  1. -1.  1.  1.  1. -1. -1.  1. -1. -1.  1.  1. -1. -1.  1.]
        [-1. -1.  1.  1. -1. -1. -1. -1. -1.  1. -1.  1.  1. -1. -1.  1.]
        [ 1. -1.  1.  1.  1.  1.  1.  1. -1. -1.  1. -1. -1.  1.  1.  1.]]
    """
    shape = (M, N)
    dict = random.bernoulli(key, shape=shape)
    dict = 2*promote_arg_dtypes(dict) - 1
    if normalize_atoms:
        return dict / math.sqrt(M)
    return dict


def random_onb(key, N):
    r"""
    Generates a random orthonormal basis for :math:`\mathbb{R}^N`

    Args:
        key: a PRNG key used as the random key.
        N (int): Dimension of the vector space

    Returns:
        (jax.numpy.ndarray): A random orthonormal basis for :math:`\mathbb{R}^N` of shape (N, N)

    Example:
        >>> from jax import random
        >>> import cr.sparse as crs
        >>> import cr.sparse.dict
        >>> Phi = cr.sparse.dict.random_onb(random.PRNGKey(0),4)
        >>> print(Phi)
        [[-0.382254 -0.266139  0.849797  0.246773]
        [ 0.518932 -0.068848 -0.035348  0.851305]
        [ 0.12152  -0.959138 -0.199282 -0.159919]
        [-0.754867 -0.066964 -0.486706  0.434522]]
    """
    A = random.normal(key, [N, N])
    Q,R = jnp.linalg.qr(A)
    return Q


def random_orthonormal_rows(key, M, N):
    """
    Generates a random sensing matrix with orthonormal rows

    Args:
        key: a PRNG key used as the random key.
        M (int): Number of rows of the sensing matrix 
        N (int): Number of columns of the sensing matrix

    Returns:
        (jax.numpy.ndarray): A random matrix of shape (M, N) with orthonormal rows

    Example:
        >>> from jax import random
        >>> import cr.sparse as crs
        >>> import cr.sparse.dict
        >>> Phi = cr.sparse.dict.random_orthonormal_rows(random.PRNGKey(0),2, 4)
        >>> print(Phi)
        [[-0.107175 -0.373504 -0.422407 -0.81889 ]
        [-0.769728 -0.300913  0.560666 -0.051218]]
    """
    A = random.normal(key, [N, M])
    Q,R = jnp.linalg.qr(A)
    return Q.T

def hadamard(n, dtype=int):
    """Hadamard matrices of size :math:`n \times n`
    """
    lg2 = int(math.log(n, 2))
    assert 2**lg2 == n, "n must be positive integer and a power of 2"
    H = jnp.array([[1]], dtype=dtype)
    for _ in range(0, lg2):
        H = jnp.vstack((jnp.hstack((H, H)), jnp.hstack((H, -H))))
    return H

def hadamard_basis(n):
    """A Hadamard basis
    """
    H = hadamard(n, dtype=jnp.float32)
    return H / math.sqrt(n)


def dirac_hadamard_basis(n):
    """A dictionary consisting of identity basis and hadamard bases
    """
    I = jnp.eye(n)
    H = hadamard_basis(n)
    return jnp.hstack((I, H))


def cosine_basis(N):
    """DCT Basis
    """
    n, k = jnp.ogrid[1:2*N+1:2, :N]
    D = 2 * jnp.cos(jnp.pi/(2*N) * n * k)
    D = normalize_l2_cw(D)
    return D.T

def dirac_cosine_basis(n):
    """A dictionary consisting of identity and DCT bases
    """
    I = jnp.eye(n)
    H = cosine_basis(n)
    return jnp.hstack((I, H))

def dirac_hadamard_cosine_basis(n):
    """A dictionary consisting of identity, Hadamard and DCT bases
    """
    I = jnp.eye(n)
    H = hadamard_basis(n)
    D = cosine_basis(n)
    return jnp.hstack((I, H, D))


def fourier_basis(n):
    """Fourier basis
    """
    F = scipy.linalg.dft(n) / math.sqrt(n)
    # From numpy to jax
    F = jnp.array(F)
    # Perform conjugate transpose
    F = hermitian(F)
    return F
