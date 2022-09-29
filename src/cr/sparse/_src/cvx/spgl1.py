# Copyright 2022 CR-Suite Development Team
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
JAX based implementation of the SPG-L1 algorithm.


References

* E. van den Berg and M. P. Friedlander, "Probing the Pareto frontier
  for basis pursuit solutions", SIAM J. on Scientific Computing,
  31(2):890-912. (2008).
"""

import math

from typing import NamedTuple, Callable

from numpy import array_str
from jax import lax, jit, device_get
import jax.numpy as jnp
from jax.numpy.linalg import norm

import cr.nimble as crn

############################################################################
#  Constants
############################################################################

EPS = jnp.finfo(jnp.float32).eps


############################################################################
#  Data Types for this module
############################################################################

class SPGL1Options(NamedTuple):
    """Options for the SPGL1 algorithm
    """
    bp_tol: float = 1e-6
    "Tolerance for basis pursuit solution"
    ls_tol: float = 1e-6
    "Tolerance for least squares solution"
    opt_tol: float = 1e-4
    "Optimality tolerance"
    dec_tol: float = 1e-4
    "Required relative change in primal objective for Newton steps"
    gamma: float = 1e-4
    "Sufficient decrease parameter"

    alpha_min: float = 1e-16
    "Minimum spectral step"
    alpha_max: float = 1e5
    "Maximum spectral step"
    memory: int = 10
    "Number of past objective values to be retained"
    max_matvec: int = jnp.inf
    "Maximum number of A x and A^T x to be computed"
    max_iters: int = 100


class SPGL1State(NamedTuple):
    """Solution state of the SPGL1 algorithm
    """
    x: jnp.ndarray
    "Solution vector"
    g : jnp.ndarray
    "Gradient vector"
    r : jnp.ndarray
    "residual vector"
    f_past: jnp.ndarray
    "Past function values"
    r_norm: float
    "Residual norm"
    r_gap: float
    "Relative duality gap"
    alpha: float
    "Step size in the current iteration"
    alpha_next: float
    "Step size for the next iteration"
    # counters
    iterations: int
    n_times: int
    "Number of multiplications with A"
    n_trans: int
    "Number of multiplications with A^T"
    n_newton: int
    "Number of newton steps"
    n_ls_iters: int
    "Number of line search iterations in the current iteration"

    def __str__(self):
        """Returns the string representation of the state
        """
        s = []
        f_val = self.f_past[0]
        g_norm = norm(self.g)
        for x in [
            f'[{self.iterations}] ',
            f'f_val:{f_val:.3f} r_gap: {self.r_gap:.3f}',
            f'g_norm:{g_norm:.3f} r_norm: {self.r_norm:.3f}',
            f'lsi:{self.n_ls_iters}, alpha: {self.alpha:.3f}, alpha_n: {self.alpha_next:.3f}',
            # f'x: {format(self.x)}',
            # f'g: {format(self.g)}',
            ]:
            s.append(x.rstrip())
        return u' '.join(s)


############################################################################
#  L1-Ball Projections
############################################################################

def _project_to_l1_ball(x, q):
    """Projects a vector inside an l1 norm ball
    """
    # sort the absolute values in descending order
    u = jnp.sort(jnp.abs(x))[::-1]
    # compute the cumulative sums
    cu = jnp.cumsum(u)
    # find the index where the cumulative sum is below the threshold
    cu_diff = cu - q
    u_scaled = u*jnp.arange(1, 1+len(u))
    flags = cu_diff > u_scaled
    K = jnp.argmax(flags)
    K = jnp.where(K == 0, len(flags), K)
    # compute the shrinkage threshold
    kappa = (cu[K-1] - q)/K
    # perform shrinkage
    return jnp.maximum(0, x - kappa) + jnp.minimum(0, x + kappa)


def project_to_l1_ball(x, q=1.):
    """Projects a vector inside an l1 norm ball
    """
    x = jnp.asarray(x)
    invalid = crn.arr_l1norm(x) > q
    return lax.cond(invalid, 
        # find the shrinkage threshold and shrink
        lambda x: _project_to_l1_ball(x, q),
        # no changes necessary
        lambda x : x, 
        x)

def project_to_l1_ball_at(x, b, q=1.):
    """Projects a vector inside an l1 norm ball centered at b
    """
    x = jnp.asarray(x)
    # compute difference from center
    r = x  - b
    r = project_to_l1_ball(r, q)
    # translate to the center
    return r + b

############################################################################
#  Weighted and unweighted primal and dual norms
############################################################################

primal_norm = crn.norm_l1
dual_norm = crn.norm_linf

def weighted_primal_l1_norm(x, w):
    return crn.norm_l1(x * w)

def weighted_dual_linf_norm(x, w):
    return crn.norm_linf(x / w)

def obj_val(r):
    """ Objective value is half of squared norm of the residual
    """
    return 0.5 * jnp.vdot(r, r)

############################################################################
#  SPG-L1 Solver
############################################################################


class CurvyLineSearchState(NamedTuple):
    alpha: float
    scale: float
    x_new: jnp.ndarray
    r_new: jnp.ndarray
    d_new: jnp.ndarray
    gtd: float
    d_norm_old: float
    f_val: float
    f_lim : float
    n_iters: int
    n_safe: int

    def __str__(self):
        """Returns the string representation of the state
        """
        s = []
        x_norm = norm(self.x_new)
        r_norm = norm(self.r_new)
        d_norm = norm(self.d_new)
        for x in [
            f'n_iters: {self.n_iters}, n_safe: {self.n_safe}',
            f'alpha: {self.alpha:.4f}, scale: {self.scale:.4f}',
            f'f_val: {self.f_val:.2f}, f_lim: {self.f_lim:.2f}',
            f'x_norm: {x_norm:.2f}, r_norm: {r_norm:.2f}',
            f', d_norm:{d_norm:.2f}'
            ]:
            s.append(x.rstrip())
        return u' '.join(s)

def curvy_line_search(A, b, x, g, alpha0, f_max, proj, tau, gamma):
    """curvilinear line search
    """
    m, n = A.shape
    max_iters = 10
    n2 = math.sqrt(n)
    g_norm = norm(g) / n2


    def candidate(alpha, scale):
        x_new = proj(x - alpha * scale * g, tau)
        r_new = b - A.times(x_new)
        d_new = x_new - x
        gtd = scale * jnp.real(jnp.dot(jnp.conj(g), d_new))
        f_val = obj_val(r_new)
        f_lim = f_max + gamma * alpha * gtd
        return x_new, r_new, d_new, gtd, f_val, f_lim

    def init():
        alpha = alpha0
        scale = 1.
        x_new, r_new, d_new, gtd, f_val, f_lim = candidate(alpha, scale)
        return CurvyLineSearchState(alpha=alpha, scale=scale,
            x_new=x_new, r_new=r_new, d_new=d_new,
            gtd=gtd, d_norm_old=0.,
            f_val=f_val, f_lim=f_max,
            n_iters=1, n_safe=0)

    def next_func(state):
        alpha = state.alpha
        # reduce alpha size
        alpha /= 2.
        # check if the scale needs to be reduced
        d_norm = norm(state.d_new)
        d_norm_old = state.d_norm_old
        # check if the iterates of x are too close to each other
        too_close = jnp.abs(d_norm - d_norm_old) <= 1e-6 * d_norm
        scale = state.scale
        n_safe = state.n_safe
        scale, n_safe = lax.cond(too_close,
            lambda _:  (d_norm / g_norm / (1 << n_safe), n_safe + 1),
            lambda _: (scale, n_safe),
            None)
        x_new, r_new, d_new, gtd, f_val, f_lim = candidate(alpha, scale)       
        return CurvyLineSearchState(alpha=alpha, scale=scale,
            x_new=x_new, r_new=r_new, d_new=d_new, 
            gtd=gtd, d_norm_old=d_norm,
            f_val=f_val, f_lim=f_lim,
            n_iters=state.n_iters+1, n_safe=n_safe)

    def cond_func(state):
        # print(state)
        a = state.n_iters < max_iters
        b = state.gtd < 0
        c = state.f_val >= state.f_lim
        a = jnp.logical_and(a, b)
        a = jnp.logical_and(a, c)
        return a

    state = init()
    state = lax.while_loop(cond_func, next_func, state)
    # while cond_func(state):
    #     state = next_func(state)
    return state


def solve_lasso_from(A,
    b: jnp.ndarray, 
    tau: float, 
    x0: jnp.ndarray,
    options: SPGL1Options = SPGL1Options()):
    # shape of the linear operator
    m, n = A.shape
    alpha_min = options.alpha_min
    alpha_max = options.alpha_max
    opt_tol = options.opt_tol
    b_norm = norm(b)

    def metrics(x, g, r, f):
        # dual norm of the gradient
        g_dnorm = dual_norm(g)
        # norm of the residual
        r_norm = norm(r)
        # duality gap
        gap = jnp.dot(jnp.conj(r), r - b) + tau * g_dnorm
        # relative duality gap
        f_m = jnp.maximum(1, f)
        r_gap = jnp.abs(gap) / f_m
        return r_norm, r_gap

    def init():
        x = jnp.asarray(x0)
        x = project_to_l1_ball(x, tau)
        # initial residual
        r = b - A.times(x)
        # initial gradient
        g = -A.trans(r)
        # objective value
        f = obj_val(r)
        # prepare the memory of past function values
        f_past = jnp.full(options.memory, f)
        # projected gradient direction
        d = project_to_l1_ball(x - g, tau) - x
        # initial step length calculation
        d_norm = crn.norm_linf(d)
        alpha =  1. / d_norm
        alpha = jnp.clip(alpha, alpha_min, alpha_max)
        r_norm, r_gap = metrics(x, g, r, f)
        return SPGL1State(x=x, g=g, r=r, 
            f_past=f_past,
            r_norm=r_norm, r_gap=r_gap, alpha=alpha,
            alpha_next=alpha,
            iterations=1, n_times=2, n_trans=2,
            n_newton=0, n_ls_iters=0)

    def body_func(state):
        f_max = jnp.max(state.f_past)
        lsearch = curvy_line_search(A, b, state.x, 
            state.g, state.alpha_next, f_max, 
            project_to_l1_ball, tau,
            options.gamma)
        # new x value
        x = lsearch.x_new
        # new residual
        r = lsearch.r_new
        # new gradient
        g = -A.trans(r)
        # new function value
        f = lsearch.f_val
        # update past values
        f_past = crn.cbuf_push_left(state.f_past, f)
        r_norm, r_gap = metrics(x, g, r, f)
        s = x - state.x
        y = g - state.g
        sts = jnp.dot(jnp.conj(s), s)
        sty = jnp.dot(jnp.conj(s), y)
        alpha_next = lax.cond(sty <= 0,
            lambda _: alpha_max,
            lambda _: jnp.clip(sts / sty, alpha_min, alpha_max),
            None)
        return SPGL1State(x=x, g=g, r=r, 
            f_past=f_past,
            r_norm=r_norm, r_gap=r_gap, 
            alpha=lsearch.alpha,
            alpha_next=alpha_next,
            iterations=state.iterations+1,
            n_times=2, n_trans=2, 
            n_newton=state.n_newton,
            n_ls_iters=lsearch.n_iters)

    def cond_func(state):
        # print(state)
        a = state.iterations < options.max_iters
        b = state.r_gap > opt_tol
        c = state.r_norm >= opt_tol * b_norm
        a = jnp.logical_and(a, b)
        a = jnp.logical_and(a, c)
        return a

    state = init()
    state = lax.while_loop(cond_func, body_func, state)
    # while cond_func(state):
    #     state = body_func(state)

    return state

def solve_lasso(A,
    b: jnp.ndarray, 
    tau: float, 
    options: SPGL1Options = SPGL1Options()):
    m, n = A.shape
    x0 = jnp.zeros(n)
    return solve_lasso_from(A, b, tau, x0, options)
