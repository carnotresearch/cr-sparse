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

from typing import NamedTuple

from jax import lax, jit
import jax.numpy as jnp


class CGState(NamedTuple):
    x: jnp.ndarray
    """The solution"""
    r: jnp.ndarray
    """The residual"""
    p: jnp.ndarray
    """The conjugate direction"""
    r_norm_sqr: jnp.ndarray
    """The residual norm squared"""
    iterations: int
    """The number of iterations it took to complete"""

def solve_from(A, b, x_0, max_iters=None, res_norm_rtol=1e-4):
    """Solves the problem :math:`Ax  = b` for a symmetric positive definite :math:`A` via conjugate gradients iterations with an initial guess.
    """
    # Boyd Conjugate Gradients slide 22
    m, n = A.shape

    b_norm_sqr = b.T @ b

    max_r_norm_sqr = b_norm_sqr * (res_norm_rtol ** 2)
    if max_iters is None:
        max_iters = m

    def init():
        # Complete one iteration
        r = b - A @ x_0
        # residual energy
        r_norm_sqr = r.T @ r
        # first conjugate direction
        p = r
        return CGState(x=x_0, r=r, p=p, 
            r_norm_sqr=r_norm_sqr,
            iterations=0)

    def iteration(state):
        # individual iteration
        p = state.p
        # common term in the computation of p.T @ A @ p and residual update
        Ap = A @ p
        # x step size along the conjugate direction
        alpha = state.r_norm_sqr / (p.T @ Ap)
        # update the solution x
        x = state.x + alpha * p
        # update the residual r
        r = state.r - alpha * Ap
        # update residual energy
        rho_1 = r.T @ r
        rho_2 = state.r_norm_sqr
        # direction update step size
        beta = rho_1 / rho_2
        # compute next conjugate direction
        p = r + beta * p
        # update state
        return CGState(x=x, r=r, p=p, 
            r_norm_sqr=rho_1,
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

solve_from_jit  = jit(solve_from,
    static_argnames=("max_iters", "res_norm_rtol"))


def solve(A, b, max_iters=None, res_norm_rtol=1e-4):
    """Solves the problem :math:`Ax  = b` for a symmetric positive definite :math:`A` via conjugate gradients iterations.
    """
    x_0 = jnp.zeros(A.shape[0])
    return solve_from_jit(A, b, x_0, max_iters=max_iters, res_norm_rtol=res_norm_rtol)


solve_jit  = jit(solve,
    static_argnames=("max_iters", "res_norm_rtol"))
