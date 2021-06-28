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
This module implements algorithms from the paper
ALTERNATING DIRECTION ALGORITHMS FOR ℓ1-PROBLEMS IN COMPRESSIVE SENSING
"""

from typing import NamedTuple, List, Dict


import jax.numpy as jnp
from jax import jit, lax, vmap
norm = jnp.linalg.norm

from cr.sparse.opt import (project_to_l2_ball, 
    project_to_linf_ball,
    shrink)


class RecoveryFullSolution(NamedTuple):
    """Represents the solution of a sparse recovery problem

    Consider a sparse recovery problem :math:`y=\Phi x + e`.

    This type combines all of this information together.

    Parameters:

        x : :estimate(s) of :math:`x`
        r : residual(s) :math:`r = y - \Phi_I x_I `
        r_norm_sqr: squared norm of residual :math:`\| r \|_2^2`
        iterations: Number of iterations required for the algorithm to converge

    Note:

        The tuple can be used to solve multiple measurement vector
        problems also. In this case, each column (of individual parameters)
        represents the solution of corresponding single vector problems.
    """
    # The non-zero values
    x: jnp.ndarray
    """Solution vector"""
    r: jnp.ndarray
    """The residuals"""
    r_norm_sqr: jnp.ndarray
    """The residual norm squared"""
    iterations: int
    """The number of iterations it took to complete"""

class BPState(NamedTuple):
    x: jnp.ndarray
    u: jnp.ndarray
    r_primal: jnp.ndarray
    r_dual: jnp.ndarray
    r_norm_sqr: jnp.ndarray
    primal_obj: float
    dual_obj: float
    iterations: int



def solve_bp(Phi, y, beta=0., gamma=1., max_iters=100, tolerance=1e-2):
    """
    Solves the problem $\min \| x \|_1 \text{s.t.} \Phi x = b$

    Args:
        Phi (jax.numpy.ndarray): Sensing matrix/dictionary
        y (jax.numpy.ndarray): Signal being approximated
        beta (float): weight for the quadratic penalty term
        gamma (float): for primal variable update
        max_iters (int): maximum number of ADMM iterations

    Returns:
        RecoveryFullSolution: Solution vector $x$ and residual $r$

    This function implements eq 2.29 of the paper.
    """
    m, n = Phi.shape
    terminated = False
    x_norm = 0.
    r_primal_norm = 0.
    r_dual_norm = 0.
    # squared norm of the signal
    y_norm_sqr = y.T @ y
    max_r_norm_sqr = y_norm_sqr * (tolerance ** 2)
    y_max  = norm(y, 'inf')
    if y_max < tolerance:
        # TODO handle special case
        pass
    # scale the problem
    y = y / y_max

    if beta == 0:
        # initialize beta
        beta = jnp.mean(jnp.abs(y))

    def init():
        # initial estimate of the solution
        x  = Phi.T @ y
        u = jnp.zeros(n)
        # primal residual
        r_primal = y - Phi @ x
        # dual residual
        r_dual = jnp.zeros(n)
        r_norm_sqr = r_primal.T @ r_primal
        primal_obj = jnp.sum(jnp.abs(x))
        # update dual objective
        dual_obj = 0.

        # initial state
        return BPState(x=x, u=u,
            r_primal=r_primal, r_dual=r_dual, r_norm_sqr=r_norm_sqr, 
            primal_obj=primal_obj, dual_obj=dual_obj,
            iterations=0)

    def iteration(state):
        # update z
        z = state.u + ( state.x/ beta)
        z = project_to_linf_ball(z)
        #  update yy
        Az = Phi @ z
        yy = Az + (state.r_primal / beta)
        # update dual residual
        u = Phi.T @ yy
        r_dual = z - u
        # update x
        x = state.x - (gamma * beta) * r_dual
        # update primal residual
        r_primal = y - Phi @ x
        # update primal objective
        primal_obj = jnp.sum(jnp.abs(x))
        # update dual objective
        dual_obj = y.T @ yy
        r_norm_sqr = r_primal.T @ r_primal
        # updatd state
        return BPState(x=x, u=u, 
            r_primal=r_primal, r_dual=r_dual, r_norm_sqr=r_norm_sqr,
            primal_obj=primal_obj, dual_obj=dual_obj,
            iterations=state.iterations+1)

    def cond(state):
        # limit on residual norm 
        # a = state.r_norm_sqr > max_r_norm_sqr
        # limit on number of iterations
        a = state.iterations < max_iters
        # gap in objectives
        gap = abs(state.primal_obj  - state.dual_obj)
        # relative gap in objective
        relative_gap = gap / jnp.abs(state.primal_obj)
        b = relative_gap > tolerance
        # combine conditions
        c = jnp.logical_and(a, b)
        # e = jnp.logical_and(c, d)
        return c

    state = lax.while_loop(cond, iteration, init())
    return RecoveryFullSolution(x=state.x, r=state.r_primal, 
        r_norm_sqr=state.r_norm_sqr, iterations=state.iterations)



