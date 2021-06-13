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
Some basic linear transformations
"""

import jax
import jax.numpy as jnp

from cr.sparse import promote_arg_dtypes
from cr.sparse import to_row_vec, to_col_vec


def householder_vec(x):
    """Computes a Householder vector for :math:`x`

    GVL4: Algorithm 5.1.1
    """
    x = promote_arg_dtypes(x)
    m = len(x)
    if m == 1:
        return jnp.array(0), jnp.array(0)
    x_1 = x[0]
    x_rest = x[1:]
    sigma = x_rest.T @ x_rest
    v = jnp.hstack((1, x_rest))

    def non_zero_sigma(v):
        mu = jnp.sqrt(x_1*x_1 + sigma)
        v_1 = jax.lax.cond(x_1 >= 0, 
            lambda _: x_1 - mu, 
            lambda _: -sigma/(x_1 + mu), 
            operand=None)
        v = v.at[0].set(v_1)
        beta = 2. * v_1 * v_1 / (sigma + v_1 * v_1)
        v = v / v_1
        return v, beta

    def zero_sigma(v):
        beta = jax.lax.cond(x_1 >= 0, lambda _: 0., lambda _: -2., operand=None)
        return v, beta

    v, beta = jax.lax.cond(sigma == 0, zero_sigma, non_zero_sigma, operand=v)
    return v , beta


def householder_matrix(x):
    """Computes a Householder refection matrix for :math:`x`
    """
    v, beta = householder_vec(x)
    return jnp.eye(len(x)) - beta * jnp.outer(v, v)


def householder_premultiply(v, beta, A):
    """Pre-multiplies a Householder reflection defined by :math:`v, beta` to a matrix A, PA
    """
    assert v.ndim == 1
    assert A.ndim == 2
    vt = to_row_vec(v)
    v = to_col_vec(v)
    return A - (beta * v) @(vt @ A)

def householder_postmultiply(v, beta, A):
    """Post-multiplies a Householder reflection defined by :math:`v, beta` to a matrix A, AP
    """
    assert v.ndim == 1
    assert A.ndim == 2
    vt = to_row_vec(v)
    v = to_col_vec(v)
    return A - (A @ v) @ (beta * vt)


def householder_vec_(x):
    """Computes a Householder vector for :math:`x`
    """
    x = promote_arg_dtypes(x)
    m = len(x)
    if m == 1:
        return jnp.array(0), jnp.array(0)
    x_1 = x[0]
    x_rest = x[1:]
    sigma = x_rest.T @ x_rest
    v = jnp.hstack((1, x_rest))
    
    if sigma == 0:
        if x_1 >= 0:
            beta = 0
        else:
            beta = -2
    else:
        mu = jnp.sqrt(x_1*x_1 + sigma)
        if x_1 <= 0:
            v = v.at[0].set(x_1 - mu)
        else:
            v = v.at[0].set(-sigma/(x_1 + mu))
        v_1 = v[0]
        beta = 2 * v_1 * v_1 / (sigma + v_1 * v_1)
        v = v / v_1
    return v , beta


def householder_ffm_jth_v_beta(A,j):
    """GVL4 EQ 5.1.4 v, beta calculation
    """
    v = A[j+1:, j]
    ms = v.T @ v
    beta = 2/(1 + ms)
    v = jnp.hstack((1, v))
    return v, beta

def householder_ffm_premultiply(A, C):
    """
    Computes Q^T C where Q is stored in its factored form in A.

    Each column j, of A contains the essential part of the j-th
    Householder vector.

    GVL4 EQ 5.1.4
    """
    m, n = A.shape
    for j in range(n):
        v, beta = householder_ffm_jth_v_beta(A, j)
        C2 = householder_premultiply(v, beta, C[j:,:])
        C = C.at[j:, :].set(C2)
    return C

def householder_ffm_backward_accum(A, k):
    """
    Computes k columns of Q from the factored form representation of Q stored in A.

    GVL4 EQ 5.1.5
    """
    m, n = A.shape
    Q = jnp.eye(m,k)
    for j in range(n-1, -1, -1):
        v, beta = householder_ffm_jth_v_beta(A, j)
        QQ = householder_premultiply(v, beta, Q[j:,j:])
        Q = Q.at[j:, j:].set(QQ)
    return Q


def householder_ffm_to_wy(A):
    """
    Computes the WY representation of Q such that Q = I_m - W Y^T from the factored form representation

    GVL4 algorithm 5.1.2
    """
    m, r = A.shape
    v, beta = householder_ffm_jth_v_beta(A, 0)
    v = to_col_vec(v)
    Y = v 
    W = beta * v
    for j in range (1, r-1):
        v, beta = householder_ffm_jth_v_beta(A, j)
        v = to_col_vec(v)
        v2 = jnp.vstack((jnp.zeros(j), v))
        z = beta * (v2  - (W @ Y[j:,:].T)@v)
        W = jnp.hstack((W, z))
        Y = jnp.hstack((Y, v2))
    return W, Y

def householder_qr_packed(A):
    """Computes the QR = A factorization of A using Householder reflections. Returns packed factorization.

    Algorithm 5.2.1
    """
    A = promote_arg_dtypes(A)
    m, n = A.shape
    assert m >= n
    for j in range(n-1):
        x = A[j:, j]
        v, beta = householder_vec(x)
        A2  = householder_premultiply(v, beta, A[j:, j:])
        A = A.at[j:, j:].set(A2)
        # place the essential part of the Householder vector
        A = A.at[j+1:,j].set(v[1:])
    return A

def householder_split_qf_r(A):
    """Splits a packed QR factorization into QF and R
    """
    # The upper triangular part is R
    R = jnp.triu(A)
    # The remaining lower triangular part of A is the factored form representation of Q
    QF = jnp.tril(A[:,:-1], -1)
    return QF, R

def householder_qr(A):
    """Computes the QR = A factorization of A using Householder reflections

    Algorithm 5.2.1
    """
    m, n = A.shape
    A = householder_qr_packed(A)
    QF , R = householder_split_qf_r(A)
    Q = householder_ffm_backward_accum(QF, n)
    return Q, R


