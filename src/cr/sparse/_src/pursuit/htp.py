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

import jax.numpy as jnp
from jax import vmap, jit, lax


from .defs import RecoverySolution, HTPState

from cr.nimble.dsp import (hard_threshold, 
    hard_threshold_sorted,
    build_signal_from_indices_and_values)

import cr.sparse.dict as crdict
import cr.sparse.lop as lop


def matrix_solve(Phi, y, K, normalized=False, step_size=None, max_iters=None, res_norm_rtol=1e-4):
    """Solves the sparse recovery problem :math:`y = \\Phi x + e` using Hard Thresholding Pursuit for matrices
    """
    ## Initialize some constants for the algorithm
    M, N = Phi.shape

    # squared norm of the signal
    y_norm_sqr = y.T @ y

    max_r_norm_sqr = y_norm_sqr * (res_norm_rtol ** 2)

    if not normalized and step_size is None:
        step_size = 0.98 / crdict.upper_frame_bound(Phi)

    if max_iters is None:
        max_iters = M

    min_iters = min(3*K, 20) 

    def compute_step_size(h, I):
        h_I = h[I]
        Phi_I = Phi[:, I]
        # Step size calculation
        Ah = Phi_I @ h_I
        mu = h_I.T @ h_I / (Ah.T @ Ah)
        return mu

    def get_step_size(h, I):
        return compute_step_size(h, I) if normalized else step_size

    def init():
        # Data for the previous approximation [r = y, x = 0]
        I_prev = jnp.arange(0, K)
        x_I_prev = jnp.zeros(K)
        r_norm_sqr_prev = y_norm_sqr
        # Assume previous estimate to be zero and conduct first iteration
        # compute the correlations of atoms with signal y
        h = Phi.T @ y
        mu = get_step_size(h, I_prev)
        # update
        x = mu * h
        # threshold
        I, x_I = hard_threshold(x, K)
        # Form the subdictionary of corresponding atoms
        Phi_I = Phi[:, I]
        # Compute new residual
        r = y - Phi_I @ x_I
        # Compute residual norm squared
        r_norm_sqr = r.T @ r
        return HTPState(x_I=x_I, I=I, r=r, r_norm_sqr=r_norm_sqr, 
            iterations=1,
            I_prev=I_prev, x_I_prev=x_I_prev, r_norm_sqr_prev=r_norm_sqr_prev)

    def iteration(state):
        I_prev = state.I
        x_I_prev = state.x_I
        r_norm_sqr_prev = state.r_norm_sqr
        # compute the correlations of dictionary atoms with the residual
        h = Phi.T @ state.r
        # current approximation
        x = build_signal_from_indices_and_values(N, state.I, state.x_I)
        # Step size calculation
        mu = get_step_size(h, I_prev)
        # update
        x = x + mu * h
        # threshold
        I, x_I = hard_threshold_sorted(x, K)
        # Form the subdictionary of corresponding atoms
        Phi_I = Phi[:, I]
        # Solve least squares over the selected K indices
        x_I, _, _, _ = jnp.linalg.lstsq(Phi_I, y)
        # Compute new residual
        y_hat = Phi_I @ x_I
        r = y - y_hat
        # Compute residual norm squared
        r_norm_sqr = r.T @ r
        return HTPState(x_I=x_I, I=I, r=r, r_norm_sqr=r_norm_sqr, 
            iterations=state.iterations+1,
            I_prev=I_prev, x_I_prev=x_I_prev, r_norm_sqr_prev=r_norm_sqr_prev
            )

    def cond(state):
        # limit on residual norm 
        a = state.r_norm_sqr > max_r_norm_sqr
        # limit on number of iterations
        b = state.iterations < max_iters
        c = jnp.logical_and(a, b)
        # checking if support is still changing
        d = jnp.any(jnp.not_equal(state.I, state.I_prev))
        # consider support change only after some iterations
        d = jnp.logical_or(state.iterations < min_iters, d)
        c = jnp.logical_and(c,d)
        # overall condition
        return c

    state = lax.while_loop(cond, iteration, init())
    return RecoverySolution(x_I=state.x_I, I=state.I, r=state.r, r_norm_sqr=state.r_norm_sqr,
        iterations=state.iterations, length=Phi.shape[1])


matrix_solve_jit  = jit(matrix_solve, static_argnums=(2), 
    static_argnames=("normalized", "step_size", "max_iters", "res_norm_rtol"))




def operator_solve(Phi, y, K, normalized=False, step_size=None, max_iters=None, res_norm_rtol=1e-4):
    """Solves the sparse recovery problem :math:`y = \\Phi x + e` using Hard Thresholding Pursuit for linear operators
    """
    ## Initialize some constants for the algorithm
    M, N = Phi.shape
    trans = Phi.trans

    # squared norm of the signal
    y_norm_sqr = y.T @ y

    max_r_norm_sqr = y_norm_sqr * (res_norm_rtol ** 2)

    if not normalized and step_size is None:
        step_size = 0.98 / lop.upper_frame_bound(Phi)

    if max_iters is None:
        max_iters = M

    min_iters = min(3*K, 20) 

    def compute_step_size(h, I):
        h_I = h[I]
        Phi_I = Phi.columns(I)
        # Step size calculation
        Ah = Phi_I @ h_I
        mu = h_I.T @ h_I / (Ah.T @ Ah)
        return mu

    def get_step_size(h, I):
        return compute_step_size(h, I) if normalized else step_size

    def init():
        # Data for the previous approximation [r = y, x = 0]
        I_prev = jnp.arange(0, K)
        x_I_prev = jnp.zeros(K)
        r_norm_sqr_prev = y_norm_sqr
        # Assume previous estimate to be zero and conduct first iteration
        # compute the correlations of atoms with signal y
        h = trans(y)
        mu = get_step_size(h, I_prev)
        # update
        x = mu * h
        # threshold
        I, x_I = hard_threshold(x, K)
        # Form the subdictionary of corresponding atoms
        Phi_I = Phi.columns(I)
        # Compute new residual
        r = y - Phi_I @ x_I
        # Compute residual norm squared
        r_norm_sqr = r.T @ r
        return HTPState(x_I=x_I, I=I, r=r, r_norm_sqr=r_norm_sqr, 
            iterations=1,
            I_prev=I_prev, x_I_prev=x_I_prev, r_norm_sqr_prev=r_norm_sqr_prev)

    def iteration(state):
        I_prev = state.I
        x_I_prev = state.x_I
        r_norm_sqr_prev = state.r_norm_sqr
        # compute the correlations of dictionary atoms with the residual
        h = trans(state.r)
        # current approximation
        x = build_signal_from_indices_and_values(N, state.I, state.x_I)
        # Step size calculation
        mu = get_step_size(h, I_prev)
        # update
        x = x + mu * h
        # threshold
        I, x_I = hard_threshold_sorted(x, K)
        # Form the subdictionary of corresponding atoms
        Phi_I = Phi.columns(I)
        # Solve least squares over the selected K indices
        x_I, r_I_norms, rank_I, s_I = jnp.linalg.lstsq(Phi_I, y)
        # Compute new residual
        y_hat = Phi_I @ x_I
        r = y - y_hat
        # Compute residual norm squared
        r_norm_sqr = r.T @ r
        return HTPState(x_I=x_I, I=I, r=r, r_norm_sqr=r_norm_sqr, 
            iterations=state.iterations+1,
            I_prev=I_prev, x_I_prev=x_I_prev, r_norm_sqr_prev=r_norm_sqr_prev
            )

    def cond(state):
        # limit on residual norm 
        a = state.r_norm_sqr > max_r_norm_sqr
        # limit on number of iterations
        b = state.iterations < max_iters
        c = jnp.logical_and(a, b)
        # checking if support is still changing
        d = jnp.any(jnp.not_equal(state.I, state.I_prev))
        # consider support change only after some iterations
        d = jnp.logical_or(state.iterations < min_iters, d)
        c = jnp.logical_and(c,d)
        # overall condition
        return c

    state = lax.while_loop(cond, iteration, init())
    return RecoverySolution(x_I=state.x_I, I=state.I, r=state.r, r_norm_sqr=state.r_norm_sqr,
        iterations=state.iterations, length=Phi.shape[1])


operator_solve_jit  = jit(operator_solve, static_argnums=(0, 2), 
    static_argnames=("normalized", "step_size", "max_iters", "res_norm_rtol"))

solve = operator_solve_jit
