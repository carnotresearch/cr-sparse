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

from jax import jit, lax

import jax.numpy as jnp
from jax.numpy.linalg import qr, norm
from jax.scipy.linalg import cholesky

import cr.nimble as cnb

def proj_zero():
    r"""Projects to the zero vector for :math:`\RR^n`
    """
    @jit
    def projector(x):
        x = jnp.asarray(x)
        return jnp.zeros_like(x)
    
    return projector


def proj_identity():
    r"""Projects on :math:`\RR^n` (x => x)
    """
    @jit
    def projector(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        return x
    
    return projector

def proj_singleton(c):

    c = jnp.asarray(c)
    c = cnb.promote_arg_dtypes(c)

    @jit
    def projector(x):
        x = jnp.asarray(x)
        return jnp.broadcast_to(c, x.shape)

    return projector


def proj_affine(A, b=0):
    """Returns a function which projects a point to the solution set A x = b
    """
    # A must be specified
    A = jnp.asarray(A)
    A = cnb.promote_arg_dtypes(A)
    # b by default is 0
    b = jnp.asarray(b)
    b = cnb.promote_arg_dtypes(b)
    # Compute the QR decomposition of A
    R = qr(A.T, 'r')
    @jit
    def projector(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        r = b - A @ x
        v = cnb.solve_UTx_b(R, r)
        w = cnb.solve_Ux_b(R, v)
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
        l = cnb.promote_arg_dtypes(l)
    if u is not None:
        u = jnp.asarray(u)
        u = cnb.promote_arg_dtypes(u)

    @jit
    def lower_bound(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        return jnp.maximum(x, l)

    @jit
    def upper_bound(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        return jnp.minimum(x, u)


    @jit
    def box_bound(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        x = jnp.maximum(x, l)
        x =  jnp.minimum(x, u)
        return x

    if l is None:
        return upper_bound

    if u is None:
        return lower_bound

    return box_bound


def proj_box_affine(l, u, a, alpha=0., tol=1e-6):
    """Projector function for the constraints l <= x <= u and a' x = alpha

    #TODO complete this
    """
    if a is None:
        raise ValueError("a is required")
    a = jnp.asarray(a)
    a = cnb.promote_arg_dtypes(a)
    n = a.size
    if l is None:
        l = jnp.full_like(a, -jnp.inf)
    if u is None:
        u = jnp.full_like(a, jnp.inf)

    @jit
    def box_bound(x):
        x = jnp.maximum(x, l)
        x =  jnp.minimum(x, u)
        return x

    def projector(x):
        # Turning points for constraints = l (l for lower)
        T1 = (x -l) / a
        # Turning points for constraints = u (u for upper)
        T2 = (x - u) / a
        T = jnp.concatenate((T1, T2))
        T = jnp.sort(T)
        lower_bound = 0
        upper_bound = 2 * n
        k = math.ceil(math.log2(2*n))
        for i in range(k):
            index = (lower_bound + upper_bound) // 2
            beta = T[index]
            # trial solution
            y = box_bound(x - beta * a)

def proj_conic():
    """Projector function for Lorentz/ice-cream cone {(x,t): \| x \|_2 \leq t}
    """
    @jit
    def proj(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        x_pre = x[:-1]
        t =  x[-1]
        abs_t = jnp.abs(t)
        norm_x = norm(x_pre)
        outside = norm_x > abs_t

        def project_inside(_):
            pre = x_pre / norm_x
            y = jnp.append(pre, 1)
            alpha = (norm_x + t) / 2
            return alpha * y

        return lax.cond(outside, 
            # bring the point inside the border
            project_inside, 
            # now we need to check the sign of t
            lambda _ : lax.cond(t > 0, 
                # the point in the upper half space, we can keep it as is
                lambda _ : x,
                # in the lower half space, the nearest point of the cone is the origin
                lambda _ : jnp.zeros_like(x),
                None)
        , None)

    return proj
