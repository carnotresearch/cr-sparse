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
Spectral projected gradient algorithm

Relevant papers:

Birgin, Ernesto G., José Mario Martínez, and Marcos Raydan. 
"Algorithm 813: SPG—software for convex-constrained optimization." 
ACM Transactions on Mathematical Software (TOMS) 27.3 (2001): 340-349.

Birgin, Ernesto G., José Mario Martínez, and Marcos Raydan. 
"Nonmonotone spectral projected gradient methods on convex sets."
SIAM Journal on Optimization 10.4 (2000): 1196-1211.
"""

from typing import NamedTuple, Callable

from numpy import array_str
from jax import lax, jit, device_get
import jax.numpy as jnp

import cr.nimble as crn
from cr.sparse.opt import SmoothFunction

EPS = jnp.finfo(jnp.float32).eps

L_MIN = 1e-10
L_MAX = 1e10
SIGMA_1 = 0.1
SIGMA_2 = 0.9
EPS_1 = 0
EPS_2 = 1e-6



def format(x):
    return array_str(device_get(x), precision=4, suppress_small=True)

class SPGOptions(NamedTuple):
    progress_tol: float = 1e-9
    "progress tolerance"
    optim_tol: float = 1e-5
    "optimality tolerance"
    suff_dec: float = 1e-4
    "sufficient decrease parameter in Armijo condition"
    max_iters: int = 1000
    "Maximum number of iterations for the solver"
    max_f_calls: int = 2000
    "Maximum number of function calls"
    memory: int = 10
    "Number of steps to look back in non-monotone Armijo condition"


class SPGState(NamedTuple):
    x: jnp.ndarray
    "Solution vector"
    g : jnp.ndarray
    "Gradient vector"
    delta: float
    "gradient, direction product"
    x_prev: jnp.ndarray
    "previous solution"
    g_prev: jnp.ndarray
    "previous gradient"
    f_past: jnp.ndarray
    "Past function values"
    # counters
    iterations: int
    proj_calls: int
    f_calls: int
    g_calls: int
    # fg_calls: int

    def __str__(self):
        """Returns the string representation of the state
        """
        s = []
        for x in [
            f'iterations: {self.iterations}, f_calls: {self.f_calls}',
            f'x: {format(self.x)}',
            f'g: {format(self.g)}',
            f'x_prev: {format(self.x_prev)}',
            f'g_prev: {format(self.g_prev)}',
            f'f_past: {format(self.f_past)}',
            f'delta: {self.delta}',
            ]:
            s.append(x.rstrip())
        return u'\n'.join(s)

class LineSearchState(NamedTuple):
    alpha: float
    x_new: jnp.ndarray
    f_val: float
    f_lim: float
    f_calls: int

def line_search(func, x, d, f_max, f_last, gamma, delta):
    gd = gamma * delta

    def init():
        x_new = x + d
        f_val = func(x_new)
        f_lim = f_max + gd
        return LineSearchState(alpha=1., x_new=x_new, 
            f_val=f_val, f_lim=f_lim, f_calls=1)

    def next_func(state):
        alpha = state.alpha
        denom = state.f_val - f_last - alpha * delta
        alpha_tmp = - 0.5 * (alpha**2) / denom
        alpha = lax.cond(
            jnp.logical_and(
                alpha_tmp >= SIGMA_1,
                alpha_tmp <= SIGMA_2 * alpha), 
            lambda _ : alpha_tmp, 
            lambda _ : alpha /2, None)
        x_new = x + alpha * d
        f_val  = func(x_new)
        f_lim = f_max + alpha * gd
        return LineSearchState(alpha=alpha, x_new=x_new, 
            f_val=f_val, f_lim=f_lim, f_calls=state.f_calls+1)

    def cond_func(state):
        return jnp.logical_and(
            state.f_val > state.f_lim,
            state.f_calls < 50)

    state = init()
    # while cond_func(state):
    #     state = next_func(state)
    state = lax.while_loop(cond_func, next_func, state)
    return state


def solve_from(smooth_f: SmoothFunction, 
    proj: Callable,
    x0: jnp.ndarray,
    options: SPGOptions = SPGOptions()):

    # sufficient decrease parameter
    gamma = options.suff_dec


    def init():
        x = proj(x0)
        g, f  = smooth_f.grad_val(x)
        d = proj(x - g) - x
        delta = jnp.dot(g, d)

        f_prev = f
        g_prev = g
        x_prev = x

        t1 = jnp.sum(jnp.abs(g))
        t = jnp.minimum(1, 1.0 / t1)

        x = x + t * d
        g, f  = smooth_f.grad_val(x)
        f_past = jnp.full(options.memory, f)

        return SPGState(x=x, g=g,
            x_prev=x_prev, g_prev=g_prev, 
            f_past=f_past,
            delta=delta,
            iterations=1,
            proj_calls=1,
            f_calls=2, g_calls=2)

    def step(state):
        x = state.x
        g = state.g
        x_prev = state.x_prev
        g_prev = state.g_prev
        y = g - g_prev
        s = x - x_prev
        lambd = (s.T @ s) / (s.T @ y + EPS)
        lambd = jnp.where(lambd < L_MIN, 1., lambd)
        lambd = jnp.where(lambd > L_MAX, 1., lambd)
        d = proj(x - lambd * g) - x
        delta = g.T @ d
        f_past = state.f_past
        f_max = jnp.max(f_past)
        f_last = f_past[0]
        lsearch = line_search(smooth_f.func, 
            x, d, 
            f_max, f_last, gamma, delta)
        x_new = lsearch.x_new
        g_new = smooth_f.grad(x_new)
        f_past = crn.cbuf_push_left(f_past, lsearch.f_val)

        return SPGState(x=x_new, g=g_new,
            delta=delta,
            x_prev=x, g_prev=g, 
            f_past=f_past,
            iterations=state.iterations+1,
            proj_calls=state.proj_calls+1,
            f_calls=state.f_calls+lsearch.f_calls,
            g_calls=state.g_calls+1)

    def cond(state):
        # if jnp.any(jnp.isnan(state.d)): return False
        opt_cond = jnp.max(jnp.abs(proj(state.x - state.g) - state.x))
        a = state.iterations < options.max_iters
        b = state.delta <= -options.progress_tol
        c = opt_cond >= options.optim_tol
        d = state.f_calls < options.max_f_calls
        res = a
        res = jnp.logical_and(res, b)
        res = jnp.logical_and(res, c)
        res = jnp.logical_and(res, d)
        # if not res: print(a,b,c,d)
        return res

    state = init()
    state = lax.while_loop(cond, step, state)
    # while cond(state):
    #     state = step(state)
    return state


solve_from_jit = jit(solve_from, static_argnums=(0,1))