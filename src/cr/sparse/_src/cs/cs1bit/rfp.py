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


"""
Renormalized fixed point iteration
"""

from typing import NamedTuple, List, Dict
import jax.numpy as jnp
from jax import jit, lax, grad, random
norm = jnp.linalg.norm

import cr.sparse as crs
from cr.sparse.opt import shrink



@jit
def rfp_random_guess(Phi):
    """Random initial guess for the solution of 1-bit compressive sensing problem
    """
    key = random.PRNGKey(0)
    M, N = Phi.shape
    x = random.normal(key, (N,))
    x = x / norm(x)
    return x

@jit
def rfp_lsqr_guess(Phi, y):
    """Returns an initial estimate to be a least square guess
    """
    x, r, rank, s = jnp.linalg.lstsq(Phi, y)
    x = x / norm(x)
    return x


# @jit
# def one_sided_quadratic(x):
#     return jnp.where(x < 0, x**2/2, 0)


# @jit
# def l1_norm(x):
#     return jnp.sum(jnp.abs(x))

# l1_norm_grad = grad(l1_norm)

# def one_sided_quadratic_cost(x):
#     return jnp.sum(one_sided_quadratic(x))

# one_sided_quadratic_cost_grad = grad(one_sided_quadratic_cost)


class RFPState(NamedTuple):
    x: jnp.ndarray
    """Current estimate"""
    x_prev: jnp.ndarray
    """Previous estimate"""
    iterations: int
    """The number of iterations completed"""


# def f_dash(x):
#     return jnp.where(x <= 0, x, 0)

def rfp(Phi, y, x0, lambda_=1., delta=0.1, inner_iters=200, outer_iters=20):
    """Solver fo 1-bit compressive sensing with renormalized fixed points iteration
    algorithm     
    """
    tau = delta / lambda_
    print(f'shrinkage threshold: {tau}')

    def rfpi_inner(x0, lambda_):
        def init_fun():
            return RFPState(x=x0, x_prev=jnp.zeros_like(x0), iterations=0)

        def body_fun(state):
            # STEP-3
            # compressive measurement based on current estimate of x
            measurement = Phi @ state.x
            # see which bits have same sign, which bits have different sign
            correlation = y * measurement
            # Only the bits which are wrong are important for further consideration
            problems = jnp.where(correlation <= 0, -correlation, 0)
            print(f'number of wrong signs: {jnp.sum(correlation < 0)}, {jnp.sum(problems)}')
            # multiply with the bit vector Y^T (f'(term))
            problems = y * problems
            # correlate with columns of Phi
            f_bar = Phi.T @ problems
            print(f'f_bar: {norm(f_bar)}')
            # STEP-4
            # Gradient Projection on Sphere Surface
            f_tilde = f_bar - jnp.vdot(f_bar,  state.x) *  state.x 
            print(f'f_tilde: {norm(f_tilde)}, {jnp.max(jnp.abs(f_tilde))}')
            # One-sided quadratic gradient descent
            h = state.x - delta * f_tilde
            #print(jnp.sum(jnp.abs(h)>0.1))
            # Shrinkage (l1 gradient descent):
            u = shrink(h, tau)
            # Normalization
            x_hat = u / norm(u)
            return RFPState(x=x_hat, x_prev=state.x, iterations=state.iterations+1)

        def cond_fun(state):
            # limit on number of iterations
            a = state.iterations < inner_iters
            diff = state.x - state.x_prev
            d_norm = norm(diff)
            #print(f'd_norm={d_norm}')
            b = d_norm > 1e-4
            return a
            return jnp.logical_and(a, b)

        state = init_fun()
        while cond_fun(state):
            state = body_fun(state)
        # state = lax.while_loop(cond, body, init())
        return state
    
    for i in range(outer_iters):
        state = rfpi_inner(x0, lambda_)
        x0 = state.x
        lambda_ = lambda_ * 1.5
    return state