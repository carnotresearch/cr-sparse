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

    Consider a sparse recovery problem :math:`b=A x + e`.

    This type combines all of this information together.

    Parameters:

        x : :estimate(s) of :math:`x`
        r : residual(s) :math:`r = b - A_I x_I `
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
    forward_count: int = 0
    """Number of times A x computed """
    adjoint_count : int = 0
    """Number of times A.T b computed """

class BPState(NamedTuple):
    x: jnp.ndarray
    """Primal variable"""
    x_prev: jnp.ndarray
    """Previous value of the primal variable"""
    z: jnp.ndarray
    """Dual variable"""
    rp: jnp.ndarray
    "Primal residual"
    rd: jnp.ndarray
    "Dual residual"
    primal_objective: float
    "Primal objective function value"
    dual_objective: float
    "Dual objective function value"
    iterations: int
    """Number of iterations"""
    forward_count: int = 0
    """Number of times A x computed """
    adjoint_count : int = 0
    """Number of times A.T b computed """



def solve_bp(A, b, beta=0., gamma=1., max_iters=100, tolerance=1e-2):
    """
    Solves the problem $\min \| x \|_1 \text{s.t.} \A x = b$

    Args:
        A (jax.numpy.ndarray): Sensing matrix/dictionary
        b (jax.numpy.ndarray): Signal being approximated
        beta (float): weight for the quadratic penalty term
        gamma (float): for primal variable update
        max_iters (int): maximum number of ADMM iterations

    Returns:
        RecoveryFullSolution: Solution vector $x$ and residual $r$

    This function implements eq 2.29 of the paper.
    """
    m, n = A.shape
    x_norm = 0.
    rp_norm = 0.
    rd_norm = 0.
    # squared norm of the signal
    y_norm_sqr = b.T @ b
    max_r_norm_sqr = y_norm_sqr * (tolerance ** 2)
    b_max  = norm(b, ord=jnp.inf)
    if b_max < tolerance:
        # TODO handle special case
        pass
    # scale the problem
    b = b / b_max

    if beta == 0:
        # initialize beta
        beta = jnp.mean(jnp.abs(b))

    def init():
        # initial estimate of the solution
        x  = A.T @ b
        u = jnp.zeros(n)
        # primal residual
        rp = b - A @ x
        # dual residual
        rd = jnp.zeros(n)
        r_norm_sqr = rp.T @ rp
        primal_objective = jnp.sum(jnp.abs(x))
        # update dual objective
        dual_objective = 0.

        # initial state
        return BPState(x=x, u=u,
            rp=rp, rd=rd, r_norm_sqr=r_norm_sqr, 
            primal_objective=primal_objective, dual_objective=dual_objective,
            iterations=0)

    def iteration(state):
        # update z
        z = state.u + ( state.x/ beta)
        z = project_to_linf_ball(z)
        #  update yy
        Az = A @ z
        yy = Az + (state.rp / beta)
        # update dual residual
        u = A.T @ yy
        rd = z - u
        # update x
        x = state.x - (gamma * beta) * rd
        # update primal residual
        rp = b - A @ x
        # update primal objective
        primal_objective = jnp.sum(jnp.abs(x))
        # update dual objective
        dual_objective = b.T @ yy
        r_norm_sqr = rp.T @ rp
        # updatd state
        return BPState(x=x, u=u, 
            rp=rp, rd=rd, r_norm_sqr=r_norm_sqr,
            primal_objective=primal_objective, dual_objective=dual_objective,
            iterations=state.iterations+1)

    def cond(state):
        # limit on residual norm 
        # a = state.r_norm_sqr > max_r_norm_sqr
        # limit on number of iterations
        a = state.iterations < max_iters
        # gap in objectives
        gap = abs(state.primal_objective  - state.dual_objective)
        # relative gap in objective
        relative_gap = gap / jnp.abs(state.primal_objective)
        b = relative_gap > tolerance
        # combine conditions
        c = jnp.logical_and(a, b)
        # e = jnp.logical_and(c, d)
        return c

    state = lax.while_loop(cond, iteration, init())
    return RecoveryFullSolution(x=b_max*state.x, r=b_max*state.rp, 
        r_norm_sqr=b_max*b_max*state.r_norm_sqr, iterations=state.iterations)



def solve_l1_l2(A, b, x0, z0, rho, gamma, tolerance, max_iters):
    """
    Solves the problem :math:`\min \| x \|_1  + \frac{1}{2 \rho} \| A x - b \|_2^2`

    Args:
        A (jax.numpy.ndarray): Sensing matrix/dictionary
        b (jax.numpy.ndarray): Signal being approximated
        rho (float): weight for the quadratic penalty term
        x0 (jax.numpy.ndarray): Initial value of solutiion (primary variable) :math:`x`
        z0 (jax.numpy.ndarray): Initial value of dual variable :math:`z`
        mu (float): term for the Augmented Lagrangian penalty
        gamma (float): ADMM update parameter for :math:`x`
        max_iters (int): maximum number of ADMM iterations

    Returns:
        RecoveryFullSolution: Solution vector $x$ and residual $r$

    This function implements eq 2.25 of the paper.
    """
    b_max  = norm(b, ord=jnp.inf)

    # scale data and model parameters
    b = b / b_max
    rho = rho / b_max
    b_norm = norm(b)
    #print(f'b_max: {b_max}')

    mu = jnp.mean(jnp.abs(b))
    mu_orig = mu
    rho_by_mu = rho / mu
    rho_by_mu_p1 = rho_by_mu + 1
    b_by_mu = b  / mu

    #print(f'mu: {mu}, rho_by_mu: {rho_by_mu}, rho_by_mu_p1: {rho_by_mu_p1}')

    def init():
        x = x0
        z = z0 
        # primal residual
        rp = b - A @ x
        # dual residual
        rd = - A.T @ b
        rp_norm_sqr = rp.T @ rp
        primal_objective = jnp.sum(jnp.abs(x)) + (0.5 / rho) * rp_norm_sqr
        # update dual objective
        dual_objective = 0.

        # initial state
        return BPState(x=x, x_prev=jnp.zeros(x.shape), z=z,
            rp=rp, rd=rd, 
            primal_objective=primal_objective, dual_objective=dual_objective,
            iterations=0, forward_count=1, adjoint_count=1)

    def iteration(state):
        # update y
        x_by_mu = state.x / mu
        y = A @ (state.z - x_by_mu) + b_by_mu
        y = y / rho_by_mu_p1
        Aty = A.T @ y
        #print(f'\n[{state.iterations+1}]', end='')
        #print(state.x[0:5])
        #print(y[0:5])
        # update z
        z = Aty + x_by_mu
        z = project_to_linf_ball(z)
        forward_count = state.forward_count + 1
        adjoint_count = state.adjoint_count + 1

        # dual residual
        rd  = z - Aty
        
        # update x
        x = state.x - (gamma*mu) * rd
        
        # primal residual
        rp = b - A @ x
        forward_count += 1
        
        # primal resdiual norm squared
        rp_norm_sqr = rp.T @ rp
        # y norm squared
        y_norm_sqr = y.T @ y

        # primary objective
        primal_objective = jnp.sum(jnp.abs(x)) + (0.5 / rho) * rp_norm_sqr
        
        # dual objective
        dual_objective = b.T @ y - (0.5 * rho) * y_norm_sqr
        
        # updatd state
        return BPState(x=x, x_prev=state.x, z=z,
            rp=rp, rd=rd,
            primal_objective=primal_objective, dual_objective=dual_objective,
            iterations=state.iterations+1, forward_count=forward_count, adjoint_count=adjoint_count)

    def double_iteration(state):
        state = iteration(state)
        return iteration(state)

    def cond(state):
        """
        Stopping condition:
        - Either relative change in x should be within tolerance.
        - Or both relative duality gap and relative dual norm should be within tolerance.
        """
        q  = 0.1
        # limit on number of iterations
        condition = state.iterations < max_iters

        # x norm
        x_norm = norm(state.x)
        # relative change in x norm
        x_relative_change = norm(state.x - state.x_prev) / x_norm
        # condition on relative change in x norm
        condition = jnp.logical_and(condition, x_relative_change > tolerance * (1 - q))

        # condition on dual residual norm
        rel_rd = norm(state.rd) / norm(state.z)

        # duality gap
        duality_gap = jnp.abs(state.dual_objective - state.primal_objective)
        # relative duality gap
        relative_gap = duality_gap / state.primal_objective

        # if either duality gap or dual res norm are beyond tolerance, we continue 
        rd_gap_cond = jnp.logical_or(relative_gap > tolerance, rel_rd > tolerance)

        # combined condition
        condition = jnp.logical_and(condition, rd_gap_cond)
        #print(f'[{state.iterations:02d}] x_norm: {x_norm:.3f}, rel:{x_relative_change:.2e} ' + 
        #    f'rel_rd {rel_rd:.2e} p_obj: {state.primal_objective:.1e}, d_obj: {state.dual_objective:.1e} relative_gap {relative_gap:.1e}')
 
        return condition

    # state = init()
    # while cond(state):
    #     state = double_iteration(state)
    state = lax.while_loop(cond, double_iteration, init())
    r_norm_sqr = state.rp.T @ state.rp
    return RecoveryFullSolution(x=b_max*state.x, r=b_max*state.rp, 
        r_norm_sqr=b_max*b_max*r_norm_sqr, iterations=state.iterations,
        forward_count=state.forward_count, adjoint_count=state.adjoint_count)


solve_l1_l2_jit = jit(solve_l1_l2, static_argnums=(4,5,6,7))



def solve(A, b, x0=None, z0=None, rho=0.01, gamma=1.0, tolerance=5e-3, max_iters=9999, jit=True):
    m = b.shape[0]
    Atb = A.T @ b
    n = Atb.shape[0]
    b_max  = norm(b, ord=jnp.inf)
    atb_max = norm(Atb, ord=jnp.inf)
    zero_solution = atb_max <= rho
    forward_count = 0
    adjoint_count = 1
    if zero_solution:
        x = jnp.zeros(n)
        return RecoveryFullSolution(x=x, iterations=0, 
            forward_count=forward_count, adjoint_count=adjoint_count)
    if x0 is None:
        x0 = Atb / b_max

    if z0 is None:
        z0 = jnp.zeros(n)

    if jit:
        if rho > 0:
            # It's an l1-l2 problem
            return solve_l1_l2_jit(A, b, x0, z0, rho, gamma, tolerance, max_iters)
        raise NotImplemented
    else:
        if rho > 0:
            # It's an l1-l2 problem
            return solve_l1_l2(A, b, x0, z0, rho, gamma, tolerance, max_iters)
        raise NotImplemented
