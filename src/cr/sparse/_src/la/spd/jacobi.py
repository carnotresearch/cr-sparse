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
Solve a symmatric positive definite system :math:`A x = b`
using Jacobi iterations
"""

from typing import NamedTuple

from jax import lax, jit
import jax.numpy as jnp


class State(NamedTuple):
    x: jnp.ndarray
    """The solution"""
    e_norm_sqr: jnp.ndarray
    """The norm squared of the error between successive x approximations"""
    iterations: int
    """The number of iterations it took to complete"""

def solve(A, b, max_iters=None, res_norm_rtol=1e-4):
    """Solves the problem :math:`Ax  = b` for a symmetric positive definite :math:`A` via conjugate gradients iterations
    """
    # Boyd Conjugate Gradients slide 22
    m, n = A.shape

    # Get the diagonal elements
    D = jnp.diag(A)
    # Invert them
    Dinv = 1 / D
    # Set the diagonal elements to 0 to get E
    E = A.at[jnp.diag_indices(m)].set(0)
    # Compute B = D^{-1} E
    B = -jnp.multiply(Dinv[:, None], E)
    # Compute z D^{-1} b 
    z = jnp.multiply(Dinv, b)

    b_norm_sqr = b.T @ b

    max_e_norm_sqr = b_norm_sqr * (res_norm_rtol ** 2)
    if max_iters is None:
        max_iters = 500

    def init():
        x = z
        e_norm_sqr = x.T @ x
        return State(x=x,
            e_norm_sqr=e_norm_sqr,
            iterations=1)

    def iteration(state):
        # update the solution x
        x = B @ state.x + z
        # update the residual r
        r = x - state.x
        e_norm_sqr = r.T @ r
        # update state
        return State(x=x,
            e_norm_sqr=e_norm_sqr,
            iterations=state.iterations+1)

    def cond(state):
        # limit on residual norm 
        a = state.e_norm_sqr > max_e_norm_sqr
        # limit on number of iterations
        b = state.iterations < max_iters
        c = jnp.logical_and(a, b)
        return c

    state = lax.while_loop(cond, iteration, init())
    return state

solve_jit  = jit(solve,
    static_argnames=("max_iters", "res_norm_rtol"))
