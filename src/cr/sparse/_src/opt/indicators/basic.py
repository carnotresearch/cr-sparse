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


from jax import jit

import jax.numpy as jnp
from jax.numpy.linalg import qr, norm

import cr.sparse as crs

def indicator_zero():

    @jit
    def indicator(x):
        is_nonzero = jnp.any(x != 0)
        return jnp.where(is_nonzero, jnp.inf, 0)
    
    return indicator


def indicator_singleton(c):

    @jit
    def indicator(x):
        is_nonzero = jnp.any(x - c != 0)
        return jnp.where(is_nonzero, jnp.inf, 0)

    return indicator


def indicator_affine(A, b):
    """Returns an indicator function for the linear system A x = b
    """
    # R = qr(A.T, 'r')

    @jit
    def indicator(x):
        # compute the residual
        r = A @ x - b
        # compute the strength of residual
        strength = norm(r) / norm(b)
        return jnp.where(strength > 1e-10, jnp.inf, 0)

    return indicator


def indicator_box(l=None, u=None):
    """Indicator function for element-wise lower and upper bounds
    """
    if l is None and u is None:
        raise ValueError("At least lower or upper bound must be defined.")
    if l is not None:
        l = jnp.asarray(l)
        l = crs.promote_arg_dtypes(l)
    if u is not None:
        u = jnp.asarray(u)
        u = crs.promote_arg_dtypes(u)

    @jit
    def lower_bound(x):
        is_invalid = jnp.any(x < l)
        return jnp.where(is_invalid, jnp.inf, 0)

    @jit
    def upper_bound(x):
        is_invalid = jnp.any(x > u)
        return jnp.where(is_invalid, jnp.inf, 0)


    @jit
    def box_bound(x):
        is_invalid = jnp.logical_or(jnp.any(x < l), jnp.any(x > u))
        return jnp.where(is_invalid, jnp.inf, 0)

    if l is None:
        return upper_bound

    if u is None:
        return lower_bound

    return box_bound

