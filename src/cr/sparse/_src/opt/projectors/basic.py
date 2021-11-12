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
from jax.scipy.linalg import cholesky

import cr.sparse as crs
import cr.sparse.la as crla

def proj_zero():

    @jit
    def projector(x):
        return jnp.zeros_like(x)
    
    return projector

def proj_singleton(c):

    @jit
    def projector(x):
        return c

    return projector


def proj_affine(A, b):
    """Returns a function which projects a point to the solution set A x = b
    """
    R = qr(A.T, 'r')

    @jit
    def projector(x):
        r = b - A @ x
        v = crla.solve_UTx_b(R, r)
        w = crla.solve_Ux_b(R, v)
        h = A.T @ w
        return x + h

    return projector


def proj_box(l=None, u=None):
    """Projector function for element-wise lower and upper bounds
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
        return jnp.maximum(x, l)

    @jit
    def upper_bound(x):
        return jnp.minimum(x, u)


    @jit
    def box_bound(x):
        x = jnp.maximum(x, l)
        x =  jnp.minimum(x, u)
        return x

    if l is None:
        return upper_bound

    if u is None:
        return lower_bound

    return box_bound
