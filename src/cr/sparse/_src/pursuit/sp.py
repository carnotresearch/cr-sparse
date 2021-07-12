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


import jax.numpy as jnp
from jax import vmap, jit, lax


from .defs import RecoverySolution

from cr.sparse import largest_indices



def matrix_solve(Phi, y, K, max_iters=None, res_norm_rtol=1e-4):
    """Solves the sparse recovery problem :math:`y = \Phi x + e` using Subspace Pursuit for matrices
    """
    ## Initialize some constants for the algorithm
    M, N = Phi.shape
    # squared norm of the signal
    y_norm_sqr = y.T @ y

    max_r_norm_sqr = y_norm_sqr * (res_norm_rtol ** 2) 

    if max_iters is None:
        max_iters = M 

    def init():
        # compute the correlations of atoms with signal y
        h = Phi.T @ y
        # Pick largest K indices [this is first iteration]
        I = largest_indices(h, K)
        # Pick corresponding atoms to form the K wide subdictionary
        Phi_I = Phi[:, I]
        # Solve least squares over the selected indices
        x_I, r_I_norms, rank_I, s_I = jnp.linalg.lstsq(Phi_I, y)
        # Compute new residual
        r = y - Phi_I @ x_I
        # Compute residual norm squared
        r_norm_sqr = r.T @ r
        # Assemble the algorithm state at the end of first iteration
        return RecoverySolution(x_I=x_I, I=I, r=r, r_norm_sqr=r_norm_sqr, iterations=1)

    def body(state):
        # compute the correlations of dictionary atoms with the residual
        h = Phi.T @ state.r
        # Ignore the previously selected atoms
        h = h.at[state.I].set(0)
        # Pick largest K indices
        I_new = largest_indices(h, K)
        # Combine with previous K indices to form a set of 2K indices
        I_2k = jnp.hstack((state.I, I_new))
        # Pick corresponding atoms to form the 2K wide subdictionary
        Phi_2I = Phi[:, I_2k]
        # Solve least squares over the selected 2K indices
        x_p, r_p_norms, rank_p, s_p = jnp.linalg.lstsq(Phi_2I, y)
        # pick the K largest indices
        Ia = largest_indices(x_p, K)
        # Identify indices for corresponding atoms
        I = I_2k[Ia]
        # TODO consider how we can exploit the guess for x_I
        # # Corresponding non-zero entries in the sparse approximation
        # x_I = x_p[Ia]
        # Form the subdictionary of corresponding atoms
        Phi_I = Phi[:, I]
        # Solve least squares over the selected K indices
        x_I, r_I_norms, rank_I, s_I = jnp.linalg.lstsq(Phi_I, y)
        # Compute new residual
        r = y - Phi_I @ x_I
        # Compute residual norm squared
        r_norm_sqr = r.T @ r
        return RecoverySolution(x_I=x_I, I=I, r=r, r_norm_sqr=r_norm_sqr, iterations=state.iterations+1)

    def cond(state):
        # limit on residual norm 
        a = state.r_norm_sqr > max_r_norm_sqr
        # limit on number of iterations
        b = state.iterations < max_iters
        c = jnp.logical_and(a, b)
        return c

    state = lax.while_loop(cond, body, init())
    return state

matrix_solve_jit = jit(matrix_solve, static_argnums=(2), static_argnames=("step_size", "max_iters", "res_norm_rtol"))


def operator_solve(Phi, y, K, max_iters=None, res_norm_rtol=1e-4):
    """Solves the sparse recovery problem :math:`y = \Phi x + e` using Subspace Pursuit for linear operators
    """
    trans = Phi.trans
    ## Initialize some constants for the algorithm
    M = y.shape[0]
    # squared norm of the signal
    y_norm_sqr = y.T @ y

    max_r_norm_sqr = y_norm_sqr * (res_norm_rtol ** 2) 

    if max_iters is None:
        max_iters = M 

    def init():
        # compute the correlations of atoms with signal y
        h = trans(y)
        # Pick largest K indices [this is first iteration]
        I = largest_indices(h, K)
        # Pick corresponding atoms to form the K wide subdictionary
        Phi_I = Phi.columns(I)
        # Solve least squares over the selected indices
        x_I, r_I_norms, rank_I, s_I = jnp.linalg.lstsq(Phi_I, y)
        # Compute new residual
        r = y - Phi_I @ x_I
        # Compute residual norm squared
        r_norm_sqr = r.T @ r
        # Assemble the algorithm state at the end of first iteration
        return RecoverySolution(x_I=x_I, I=I, r=r, r_norm_sqr=r_norm_sqr, iterations=1)

    def body(state):
        # compute the correlations of dictionary atoms with the residual
        h = trans(state.r)
        # Ignore the previously selected atoms
        h = h.at[state.I].set(0)
        # Pick largest K indices
        I_new = largest_indices(h, K)
        # Combine with previous K indices to form a set of 2K indices
        I_2k = jnp.hstack((state.I, I_new))
        # Pick corresponding atoms to form the 2K wide subdictionary
        Phi_2I = Phi.columns(I_2k)
        # Solve least squares over the selected 2K indices
        x_p, r_p_norms, rank_p, s_p = jnp.linalg.lstsq(Phi_2I, y)
        # pick the K largest indices
        Ia = largest_indices(x_p, K)
        # Identify indices for corresponding atoms
        I = I_2k[Ia]
        # TODO consider how we can exploit the guess for x_I
        # # Corresponding non-zero entries in the sparse approximation
        # x_I = x_p[Ia]
        # Form the subdictionary of corresponding atoms
        Phi_I = Phi.columns(I)
        # Solve least squares over the selected K indices
        x_I, r_I_norms, rank_I, s_I = jnp.linalg.lstsq(Phi_I, y)
        # Compute new residual
        r = y - Phi_I @ x_I
        # Compute residual norm squared
        r_norm_sqr = r.T @ r
        return RecoverySolution(x_I=x_I, I=I, r=r, r_norm_sqr=r_norm_sqr, iterations=state.iterations+1)

    def cond(state):
        # limit on residual norm 
        a = state.r_norm_sqr > max_r_norm_sqr
        # limit on number of iterations
        b = state.iterations < max_iters
        c = jnp.logical_and(a, b)
        return c

    state = lax.while_loop(cond, body, init())
    return state

operator_solve_jit = jit(operator_solve, static_argnums=(0, 2), static_argnames=("step_size", "max_iters", "res_norm_rtol"))

solve = operator_solve_jit
