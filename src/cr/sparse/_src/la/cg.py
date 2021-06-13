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

from typing import NamedTuple

from jax import lax, jit
import jax.numpy as jnp


class CGState(NamedTuple):
    x: jnp.ndarray
    """The solution"""
    r: jnp.ndarray
    """The residual"""
    p: jnp.ndarray
    """The direction"""
    r_norm_sqr: jnp.ndarray
    """The residual norm squared"""
    prev_r_norm_sqr: jnp.ndarray
    """The residual norm squared"""
    iterations: int
    """The number of iterations it took to complete"""

def solve(A, b, max_iters=None, res_norm_rtol=1e-4):
    """Solves the problem Ax  = b via conjugate gradients iterations
    """
    # Boyd Conjugate Gradients slide 22
    m, n = A.shape

    b_norm_sqr = b.T @ b

    max_r_norm_sqr = b_norm_sqr * (res_norm_rtol ** 2)
    if max_iters is None:
        max_iters = m

    def init():
        # Complete one iteration
        # Steps of first iteration have been simplified.
        # r = b
        # r_norm_sqr = b_norm_sqr
        # p = r
        p = b
        w = A @ p
        alpha = b_norm_sqr / (p.T @ w)
        x = alpha * p
        r = b - alpha * w
        r_norm_sqr = r.T @ r
        return CGState(x=x, r=r, p=p, 
            r_norm_sqr=r_norm_sqr, prev_r_norm_sqr=b_norm_sqr,
            iterations=1)

    def iteration(state):
        # individual iteration
        rho_1 = state.r_norm_sqr
        rho_2 = state.prev_r_norm_sqr
        # compute p from previous r and p values
        p = state.r + (rho_1 / rho_2) * state.p
        w = A @ p
        alpha = rho_1 / (p.T @ w)
        # update the solution x
        x = state.x + alpha * p
        # update the residual r
        r = b - alpha * w
        r_norm_sqr = r.T @ r
        # update state
        return CGState(x=x, r=r, p=p, 
            r_norm_sqr=r_norm_sqr, prev_r_norm_sqr=rho_1,
            iterations=state.iterations+1)

    def cond(state):
        # limit on residual norm 
        a = state.r_norm_sqr > max_r_norm_sqr
        # limit on number of iterations
        b = state.iterations < max_iters
        c = jnp.logical_and(a, b)
        return c 

    state = lax.while_loop(cond, iteration, init())
    return state

solve_jit  = jit(solve,
    static_argnames=("max_iters", "res_norm_rtol"))
