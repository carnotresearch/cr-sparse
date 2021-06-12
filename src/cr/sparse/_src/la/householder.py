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
from cr.sparse import to_row_vec


def householder_vec(x):
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
    """Computes a Householder refection operator matrix for :math:`x`
    """
    v, beta = householder_vec(x)
    return jnp.eye(len(x)) - beta * jnp.outer(v, v)


def householder_premultiply(v, beta, A):
    """Pre-multiplies a Householder reflection defined by :math:`v, beta` to a matrix A
    """
    vt = to_row_vec(v)
    return A - (beta * v) @(vt @ A)

def householder_postmultiply(v, beta, A):
    """Post-multiplies a Householder reflection defined by :math:`v, beta` to a matrix A
    """
    vt = to_row_vec(v)
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
