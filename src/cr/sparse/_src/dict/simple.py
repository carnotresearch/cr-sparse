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
    """
    shape = (N, D)
    dict = random.normal(key, shape)
    if normalize_atoms:
        dict = normalize_l2_cw(dict)
    else:
        sigma = math.sqrt(N)
        dict = dict / sigma
    return dict

def rademacher_mtx(key, M, N):
    """A dictionary/sensing matrix where entries are drawn independently from Rademacher distribution.
    """
    shape = (M, N)
    dict = random.bernoulli(key, shape=shape)
    dict = 2*promote_arg_dtypes(dict) - 1
    return dict / math.sqrt(M)


def random_onb(key, N):
    """
    Generates a random orthonormal basis
    """
    A = random.normal(key, [N, N])
    Q,R = jnp.linalg.qr(A)
    return Q


def random_orthonormal_rows(key, M, N):
    """
    Generates a random sensing matrix with orthonormal rows
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
    for i in range(0, lg2):
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
