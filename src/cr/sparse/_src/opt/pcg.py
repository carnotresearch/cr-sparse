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

def _identity(x):
  return x


class PCGState(NamedTuple):
    x: jnp.ndarray
    """The solution"""
    r: jnp.ndarray
    """The residual"""
    p: jnp.ndarray
    """The conjugate direction"""
    gamma: float
    """The residual norm squared"""
    iterations: int
    """The number of iterations it took to complete"""

def solve_from(A, b, x0, max_iters=20, tol=1e-4, atol=0.0, M=_identity):
    """Solves the problem :math:`Ax  = b` for a symmetric positive definite :math:`A` via preconditioned conjugate gradients iterations with an initial guess and a preconditioner.
    """
    # Boyd Conjugate Gradients slide 22

    b_norm_sqr = jnp.vdot(b, b)
    max_gamma = jnp.maximum(jnp.square(tol) * b_norm_sqr, jnp.square(atol))
    #print(f'{b_norm_sqr=}, {max_gamma=}, {max_iters=}')
    # if max_iters is None:
    #     max_iters = b.shape[0]
    def init():
        # Complete one iteration
        r0 = b - A (x0)
        # first conjugate direction
        p0 = z0 = M(r0)
        # residual energy
        gamma = jnp.dot(r0, z0).astype(float)
        return PCGState(x=x0, r=r0, p=p0, 
            gamma=gamma,
            iterations=1)

    def body(state):
        # individual iteration
        p = state.p
        # common term in the computation of p.T @ A @ p and residual update
        Ap = A(p)
        # x step size along the conjugate direction
        alpha = state.gamma / jnp.vdot(p, Ap)
        # update the solution x
        x = state.x + alpha * p
        # update the residual r
        r = state.r - alpha * Ap
        # Auxiliary variable
        z = M(r)
        # update residual energy
        gamma = jnp.vdot(r, z).astype(float)
        # direction update step size
        beta = gamma / state.gamma
        # compute next conjugate direction
        p = z + beta * p
        # update state
        return PCGState(x=x, r=r, p=p, 
            gamma=gamma,
            iterations=state.iterations+1)

    def cond(state):
        # limit on residual norm
        r = state.r
        # gamma may not have residual energy if a preconditioner is setup
        gamma = state.gamma if M is _identity else jnp.vdot(r,r)
        #print(f'{gamma=}, {max_gamma=}, {state.iterations=}') 
        # limit on number of iterations
        return (gamma > max_gamma) & (state.iterations < max_iters) 

    # state = init()
    # while cond(state):
    #     state = body(state)
    state = lax.while_loop(cond, body, init())
    return state

solve_from_jit  = jit(solve_from,
    static_argnames=("A", "max_iters", "tol", "atol", "M"))

def solve(A, b, max_iters=20, tol=1e-4, atol=0.0, M=_identity):
    """Solves the problem :math:`Ax  = b` for a symmetric positive definite :math:`A` via preconditioned conjugate gradients iterations with a preconditioner.
    """
    x0 = jnp.zeros_like(b)
    return solve_from_jit(A, b, x0, max_iters, tol, atol, M)


solve_jit  = jit(solve,
    static_argnames=("A", "max_iters", "tol", "atol", "M"))
