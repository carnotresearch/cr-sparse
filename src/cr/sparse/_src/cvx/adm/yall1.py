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

from cr.sparse import lop
from cr.sparse import RecoveryFullSolution

def project_to_box(z, w):
    ww = jnp.maximum(w, jnp.abs(z))
    factors = w / ww
    return z * factors

def project_to_real_upper_limit(z, w):
    return jnp.minimum(w, jnp.real(z))


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
    n_times: int = 0
    """Number of times A x computed """
    n_trans : int = 0
    """Number of times A.T b computed """

def solve_bp(A, b, x0, z0, w, nonneg, gamma, tolerance, max_iters):
    """
    Solves the problem :math:`\min \| x \|_1 \, \\text{s.t.}\, A x = b` using ADMM

    This function implements eq 2.29 of the paper.
    """
    times = A.times
    trans = A.trans
    mu = jnp.mean(jnp.abs(b))
    mu_orig = mu
    b_by_mu = b  / mu
    rp_norm_threshold = tolerance * norm(b)

    def init():
        # primal residual
        rp = b - times(x0)
        # dual residual
        rd = - trans(b)
        primal_objective = jnp.sum(jnp.abs(w*x0))
        # update dual objective
        dual_objective = 0.
        # initial state
        return BPState(x=x0, x_prev=jnp.zeros(x0.shape), z=z0,
            rp=rp, rd=rd, 
            primal_objective=primal_objective, dual_objective=dual_objective,
            iterations=0, n_times=1, n_trans=1)

    def iteration(state):
        # update y
        x_by_mu = state.x / mu
        y = times(state.z - x_by_mu) + b_by_mu
        Aty = trans(y)
        # update z
        z = Aty + x_by_mu
        z = jnp.where(nonneg, project_to_real_upper_limit(z, w), project_to_box(z, w))
        n_times = state.n_times + 1
        n_trans = state.n_trans + 1

        # dual residual
        rd  = z - Aty
        
        # update x
        x = state.x - (gamma*mu) * rd
        
        # primal residual
        rp = b - times(x)
        n_times += 1
        
        # primary objective
        primal_objective = jnp.sum(jnp.abs(w*x))        
        # dual objective
        dual_objective = b.T @ y

        # print(f'x[{state.iterations+1}]', end='')
        # print(x[0:6])
        # print(f'y[{state.iterations+1}]', end='')
        # print(y[0:6])
        # print(f'z[{state.iterations+1}]', end='')
        # print(z[0:6])
        
        # updatd state
        return BPState(x=x, x_prev=state.x, z=z,
            rp=rp, rd=rd,
            primal_objective=primal_objective, dual_objective=dual_objective,
            iterations=state.iterations+1, n_times=n_times, n_trans=n_trans)

    def double_iteration(state):
        state = iteration(state)
        return iteration(state)

    def cond(state):
        """
        Stopping condition:
        - Either relative change in x should be within tolerance.
        - Or both relative duality gap and relative dual norm should be within tolerance.
        """
        q = 0.1
        # limit on number of iterations
        more_iters = state.iterations < max_iters

        # x norm
        x_norm = norm(state.x)
        # relative change in x norm
        x_relative_change = norm(state.x - state.x_prev) / x_norm
        # condition on relative change in x norm
        x_unstable = x_relative_change > tolerance * (1 - q)

        # condition on dual residual norm
        rel_rd = norm(state.rd) / norm(state.z)
        d_infeasible = rel_rd > tolerance

        # duality gap
        duality_gap = jnp.abs(state.dual_objective - state.primal_objective)
        # relative duality gap
        relative_gap = duality_gap / state.primal_objective
        gap_infeasible = relative_gap > tolerance

        # primal residual norm
        rp_norm = norm(state.rp)
        # check feasibility of primal residual norm
        p_infeasible = rp_norm >= rp_norm_threshold

        # if either duality gap or dual res norm are beyond tolerance, we continue 
        condition = jnp.logical_or(gap_infeasible, d_infeasible)
        condition = jnp.logical_or(condition, p_infeasible)
        condition = jnp.logical_and(condition, more_iters)
        condition = jnp.logical_and(condition, x_unstable)

        # print(f'[{state.iterations:02d}] x_norm: {x_norm:.3f}, rel:{x_relative_change:.2e} ' + 
        #    f'rel_rd {rel_rd:.2e} rp_norm: {rp_norm:.2e} p_infeasible: {p_infeasible}' +
        #    f' p_obj: {state.primal_objective:.1e}, d_obj: {state.dual_objective:.1e} relative_gap {relative_gap:.1e}')
 
        return condition

    # state = init()
    # while cond(state):
    #     state = double_iteration(state)
    state = lax.while_loop(cond, double_iteration, init())
    return state

solve_bp_jit = jit(solve_bp, static_argnums=(0, 5,6, 7, 8))


def solve_l1_l2(A, b, x0, z0, w, nonneg, rho, gamma, tolerance, max_iters):
    """
    Solves the problem :math:`\min \| x \|_1  + \\frac{1}{2 \\rho} \| A x - b \|_2^2` using ADMM

    This function implements eq 2.25 of the paper.
    """
    times = A.times
    trans = A.trans
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
        rp = b - times(x)
        # dual residual
        rd = - trans(b)
        rp_norm_sqr = rp.T @ rp
        primal_objective = jnp.sum(jnp.abs(w*x)) + (0.5 / rho) * rp_norm_sqr
        # update dual objective
        dual_objective = 0.

        # initial state
        return BPState(x=x, x_prev=jnp.zeros(x.shape), z=z,
            rp=rp, rd=rd, 
            primal_objective=primal_objective, dual_objective=dual_objective,
            iterations=0, n_times=1, n_trans=1)

    def iteration(state):
        # update y
        x_by_mu = state.x / mu
        y = times(state.z - x_by_mu) + b_by_mu
        y = y / rho_by_mu_p1
        Aty = trans(y)
        #print(f'\n[{state.iterations+1}]', end='')
        #print(state.x[0:5])
        #print(y[0:5])
        # update z
        z = Aty + x_by_mu
        z = jnp.where(nonneg, project_to_real_upper_limit(z, w), project_to_box(z, w))
        n_times = state.n_times + 1
        n_trans = state.n_trans + 1

        # dual residual
        rd  = z - Aty
        
        # update x
        x = state.x - (gamma*mu) * rd
        
        # primal residual
        rp = b - times(x)
        n_times += 1
        
        # primal resdiual norm squared
        rp_norm_sqr = rp.T @ rp
        # y norm squared
        y_norm_sqr = y.T @ y

        # primary objective
        primal_objective = jnp.sum(jnp.abs(w*x)) + (0.5 / rho) * rp_norm_sqr
        
        # dual objective
        dual_objective = b.T @ y - (0.5 * rho) * y_norm_sqr
        
        # updatd state
        return BPState(x=x, x_prev=state.x, z=z,
            rp=rp, rd=rd,
            primal_objective=primal_objective, dual_objective=dual_objective,
            iterations=state.iterations+1, n_times=n_times, n_trans=n_trans)

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
    return state


solve_l1_l2_jit = jit(solve_l1_l2, static_argnums=(0, 5,6,7,8,9))



def solve_l1_l2con(A, b, x0, z0, w, nonneg, delta, gamma, tolerance, max_iters):
    """
    Solves the problem :math:`\min \| x \|_1  \\text{s.t.} \| A x - b \|_2 \\leq \\delta` using ADMM

    This function implements eq 2.27 of the paper.
    """
    times = A.times
    trans = A.trans
    mu = jnp.mean(jnp.abs(b))
    mu_orig = mu
    b_by_mu = b  / mu
    delta_by_mu = delta / mu
    rp_norm_threshold = delta * (1 + tolerance)

    # print(f'mu: {mu}, delta_by_mu: {delta_by_mu}, rp_norm_threshold: {rp_norm_threshold}')
    # print(f'x0', end='')
    # print(x0[0:6])

    def init():
        # primal residual
        rp = b - times(x0)
        # dual residual
        rd = - trans(b)
        primal_objective = jnp.sum(jnp.abs(w*x0))
        # update dual objective
        dual_objective = 0.
        # initial state
        return BPState(x=x0, x_prev=jnp.zeros(x0.shape), z=z0,
            rp=rp, rd=rd, 
            primal_objective=primal_objective, dual_objective=dual_objective,
            iterations=0, n_times=1, n_trans=1)

    def iteration(state):
        # update y
        x_by_mu = state.x / mu
        y = times (state.z - x_by_mu) + b_by_mu
        # subtract projection of y to l2 ball from y
        y_norm = norm(y)
        y = jnp.maximum(0, 1 - delta_by_mu / y_norm) * y
        Aty = trans(y)
        # update z
        z = Aty + x_by_mu
        z = jnp.where(nonneg, project_to_real_upper_limit(z, w), project_to_box(z, w))
        n_times = state.n_times + 1
        n_trans = state.n_trans + 1

        # dual residual
        rd  = z - Aty
        
        # update x
        x = state.x - (gamma*mu) * rd
        
        # primal residual
        rp = b - times(x)
        n_times += 1
        
        # primary objective
        primal_objective = jnp.sum(jnp.abs(w*x))        
        # dual objective
        dual_objective = b.T @ y - delta * norm(y)

        # print(f'x[{state.iterations+1}]', end='')
        # print(x[0:6])
        # print(f'y[{state.iterations+1}]', end='')
        # print(y[0:6])
        # print(f'z[{state.iterations+1}]', end='')
        # print(z[0:6])
        
        # updatd state
        return BPState(x=x, x_prev=state.x, z=z,
            rp=rp, rd=rd,
            primal_objective=primal_objective, dual_objective=dual_objective,
            iterations=state.iterations+1, n_times=n_times, n_trans=n_trans)

    def double_iteration(state):
        state = iteration(state)
        return iteration(state)

    def cond(state):
        """
        Stopping condition:
        - Either relative change in x should be within tolerance.
        - Or both relative duality gap and relative dual norm should be within tolerance.
        """
        # limit on number of iterations
        more_iters = state.iterations < max_iters

        # x norm
        x_norm = norm(state.x)
        # relative change in x norm
        x_relative_change = norm(state.x - state.x_prev) / x_norm
        # condition on relative change in x norm
        x_unstable = x_relative_change > tolerance

        # condition on dual residual norm
        rel_rd = norm(state.rd) / norm(state.z)
        d_infeasible = rel_rd > tolerance

        # duality gap
        duality_gap = jnp.abs(state.dual_objective - state.primal_objective)
        # relative duality gap
        relative_gap = duality_gap / state.primal_objective
        gap_infeasible = relative_gap > tolerance

        # primal residual norm
        rp_norm = norm(state.rp)
        # check feasibility of primal residual norm
        p_infeasible = rp_norm > rp_norm_threshold

        # if either duality gap or dual res norm are beyond tolerance, we continue 
        condition = jnp.logical_or(gap_infeasible, d_infeasible)
        condition = jnp.logical_or(condition, p_infeasible)
        condition = jnp.logical_and(condition, more_iters)
        condition = jnp.logical_and(condition, x_unstable)

        # print(f'[{state.iterations:02d}] x_norm: {x_norm:.3f}, rel:{x_relative_change:.2e} ' + 
        #    f'rel_rd {rel_rd:.2e} rp_norm: {rp_norm:.2e} p_infeasible: {p_infeasible}' +
        #    f' p_obj: {state.primal_objective:.1e}, d_obj: {state.dual_objective:.1e} relative_gap {relative_gap:.1e}')
 
        return condition

    # state = init()
    # while cond(state):
    #     state = double_iteration(state)
    state = lax.while_loop(cond, double_iteration, init())
    return state


solve_l1_l2con_jit = jit(solve_l1_l2con, static_argnums=(0, 5,6,7,8, 9))



def solve(A, b, x0=None, z0=None, W=None, weights=None, nonneg=False, rho=0., delta=0., gamma=1.0, tolerance=5e-3, max_iters=9999, jit=True):
    """Wrapper method to solve a variety of l1 minimization problems using ADMM

    Args:
        A (jax.numpy.ndarray): Sensing matrix/dictionary
        b (jax.numpy.ndarray): Signal being approximated
        x0 (jax.numpy.ndarray): Initial value of solutiion (primary variable) :math:`x`
        z0 (jax.numpy.ndarray): Initial value of dual variable :math:`z`
        nonneg (bool): Flag to indicate if values in the solution are all non-negative
        W (jax.numpy.ndarray): The sparsifying orthonormal basis such that :math:`W x` is sparse
        weights (jax.numpy.ndarray): The weights for individual entries in :math:`x`
        rho (float): weight for the quadratic penalty term
        delta (float): constraint on the residual norm
        gamma (float): ADMM update parameter for :math:`x`
        max_iters (int): maximum number of ADMM iterations

    Returns:
        RecoveryFullSolution: Solution vector :math:`x` and residual :math:`r`

    This function implements eq 2.25 of the paper.
    """
    if W:
        # change A to solve for alpha = W x
        A = A @ W    
    m = b.shape[0]
    Atb = A.trans(b)
    n_times = 0
    n_trans = 1
    n = Atb.shape[0]
    b_max  = float(norm(b, ord=jnp.inf))
    atb_max = float(norm(Atb, ord=jnp.inf))
    zero_solution = False
    if rho > 0:
        zero_solution = atb_max <= rho
    if delta > 0:
        zero_solution = norm(b) <= delta
    if zero_solution:
        x = jnp.zeros(n)
        return RecoveryFullSolution(x=x, r=b, 
            iterations=0, 
            n_times=n_times, n_trans=n_trans)
    if x0 is None:
        x0 = Atb / b_max

    if z0 is None:
        z0 = jnp.zeros(n)

    w = jnp.ones(n)
    if weights is not None:
        # make sure that the final weights are an array of size n
        w = w * weights

    # scale data and model parameters
    b = b / b_max
    if rho > 0: rho = rho / b_max
    if delta > 0: delta = delta / b_max

    if jit:
        if rho > 0:
            # It's an l1-l2 problem
            state = solve_l1_l2_jit(A, b, x0, z0, w, nonneg, rho, gamma, tolerance, max_iters)
        elif delta > 0:
            # It's an l1-l2 constrained problem BPIC
            state = solve_l1_l2con_jit(A, b, x0, z0, w, nonneg, delta, gamma, tolerance, max_iters)
        else:
            # It's a basis pursuit problem
            state = solve_bp_jit(A, b, x0, z0, w, nonneg, gamma, tolerance, max_iters)
    else:
        if rho > 0:
            # It's an l1-l2 problem BPDN
            state = solve_l1_l2(A, b, x0, z0, w, nonneg, rho, gamma, tolerance, max_iters)
        elif delta > 0:
            # It's an l1-l2 constrained problem BPIC
            state = solve_l1_l2con(A, b, x0, z0, w, nonneg, delta, gamma, tolerance, max_iters)
        else:
            # It's a basis pursuit problem
            state = solve_bp(A, b, x0, z0, w, nonneg, gamma, tolerance, max_iters)
    x = jnp.where(nonneg, jnp.maximum(0, state.x), state.x)
    if W:
        # go back from sparsifying basis to signal space
        x = W.times(x)
    return RecoveryFullSolution(x=b_max*x, r=b_max*state.rp, 
        iterations=state.iterations,
        n_times=state.n_times+n_times, 
        n_trans=state.n_trans+n_trans)
