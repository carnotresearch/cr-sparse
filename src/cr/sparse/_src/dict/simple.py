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
from functools import partial

import numpy as np
import scipy

import jax.numpy as jnp
from jax import random
from jax import jit, vmap
from jax.experimental import sparse

from cr.nimble import (normalize_l2_cw, 
    promote_arg_dtypes, hermitian,
    diag_postmultiply)
import cr.wavelets as crwt


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
        [[ 1  1  1 -1 -1  1 -1 -1  1  1  1 -1  1  1 -1  1]
        [-1  1  1  1 -1  1  1  1 -1 -1 -1  1 -1 -1 -1  1]
        [ 1 -1 -1  1 -1  1  1 -1  1 -1  1 -1 -1 -1  1 -1]
        [ 1 -1 -1  1 -1  1  1 -1  1 -1 -1  1  1  1 -1  1]
        [-1 -1 -1 -1 -1 -1 -1  1 -1 -1 -1  1 -1 -1 -1 -1]
        [-1  1 -1  1  1  1 -1 -1  1 -1 -1  1  1 -1 -1  1]
        [-1 -1  1  1 -1 -1 -1 -1 -1  1 -1  1  1 -1 -1  1]
        [ 1 -1  1  1  1  1  1  1 -1 -1  1 -1 -1  1  1  1]]
    """
    shape = (M, N)
    dict = random.bernoulli(key, shape=shape)
    dict = 2*dict - 1
    if normalize_atoms:
        return dict / math.sqrt(M)
    return dict


def sparse_binary_mtx(key, M, N, d, normalize_atoms=True, dense=False):
    """A sensing matrix where exactly d entries are 1 in each column

    Args:
        key: a PRNG key used as the random key.
        M (int): Number of rows of the sensing matrix 
        N (int): Number of columns of the sensing matrix
        d (int): Number of 1s in each column
        normalize_atoms (bool): Whether the columns of sensing matrix are normalized 
          (default True)
        dense (bool): Whether to return a dense or a sparse matrix

    Returns:
        (jax.experimental.sparse.bcoo.BCOO): A sparse binary matrix
        where each column contains exactly d ones and (M-d) zeros.

    Note:

        The resultant matrix is stored in the BCOO format.
    """
    # create keys for the N columns
    keys = random.split(key, N)
    # indices
    idx = jnp.arange(M)
    rc = lambda key : jnp.zeros(M, dtype=jnp.uint8).at[jnp.sort(random.choice(key, idx, (d, ), replace=False))].set(1)
    dict = vmap(rc, out_axes=1)(keys)
    if normalize_atoms:
        dict = dict / math.sqrt(d)
    if dense:
        return dict
    return sparse.BCOO.fromdense(dict)

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
    dg = jnp.sign(jnp.diag(R))
    # apply the random sign changes
    Q = diag_postmultiply(Q, dg)
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
    dg = jnp.sign(jnp.diag(R))
    # apply the random sign changes
    Q = diag_postmultiply(Q, dg)
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


def wavelet_basis(n, name, level=None):
    """Builds a wavelet basis for a given decomposition level

    Note:
        This function generates orthogonal bases only for orthogonal wavelets.
        For the biorthogonal wavelets, the generated basis is a basis
        but not an orthogonal basis.
    """
    wavelet = crwt.to_wavelet(name)
    rec_lo = wavelet.rec_lo
    rec_hi = wavelet.rec_hi
    mode = 'periodization'
    m = crwt.next_pow_of_2(n)
    assert m == n, f"n={n} must be a power of 2"
    # We need to verify that the level is not too high
    max_level = crwt.dwt_max_level(n, wavelet.dec_len)
    if level is None:
        level = max_level
    else:
        assert level <= max_level, f"Level too high level={level}, max_level={max_level}"

    def waverec(coefs):
        mid = coefs.shape[0] >> level
        a = coefs[:mid]
        end = mid*2
        for j in range(level):
            d = coefs[mid:end]
            a = crwt.idwt_(a, d, rec_lo, rec_hi, 'periodization')
            mid = end
            end = mid * 2
        return a
    data = jnp.eye(n)
    basis = jnp.apply_along_axis(waverec, 0, data)
    return basis
