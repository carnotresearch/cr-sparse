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


from jax import jit, grad, lax

import jax.numpy as jnp
from jax.numpy.linalg import det, cholesky, inv
import cr.nimble as cnb

from .smooth import build2, build3

def smooth_quad_matrix(P=None, q=None, r=None):
    r"""Quadratic function and its gradient :math:`f(x) = \frac{1}{2} x^T P x + \langle q, x \rangle + r`
    """
    if P is not None:
        P = jnp.asarray(P)
        P = cnb.promote_arg_dtypes(P)
    if q is not None:
        q = jnp.asarray(q)
        q = cnb.promote_arg_dtypes(q)

    @jit
    def func(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        if P is None:
            v = 0.5 * cnb.arr_rdot(x, x)
        else:
            v = 0.5 * cnb.arr_rdot(P @ x, x)
        if q is not None:
            v = v +  cnb.arr_rdot(q, x)
        if r is not None:
            v = v + r
        return v


    @jit
    def gradient(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        if P is None:
            g = x
        else:
            g = P @ x
        if q is not None:
            g = g + q
        return g

    return build2(func, gradient)


def smooth_quad_error(A, b):
    r"""Quadratic error function and its gradient :math:`f(x) = \frac{1}{2} \| A x - b \|_2^2`
    """

    @jit
    def func(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        r = A @ x - b
        return 0.5 * jnp.dot(r, r)


    @jit
    def gradient(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        r = A @ x - b
        g = r.T @ A
        return g


    @jit
    def grad_val(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        r = A @ x - b
        v = 0.5 * jnp.dot(r, r)
        g = r.T @ A
        return g, v

    return build3(func, gradient, grad_val)
