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


from .defs import RecoverySolution, CoSaMPState

from .util import largest_indices

EXTRA_FACTOR = 2


def matrix_solve(Phi, y, K, max_iters=None, res_norm_rtol=1e-4):
    r"""Solves the sparse recovery problem :math:`y = \Phi x + e` using Compressive Sampling Matching Pursuit for matrices
    """
    M = y.shape[0]
    ## Initialize some constants for the algorithm
    K2 = EXTRA_FACTOR * K
    K3 = K + K2
    # squared norm of the signal
    y_norm_sqr = y.T @ y

    max_r_norm_sqr = y_norm_sqr * (res_norm_rtol ** 2) 

    if max_iters is None:
        max_iters = M 

    min_iters = min(3*K, 20) 

    def init():
        # Data for the previous approximation [r = y, x = 0]
        I_prev = jnp.arange(0, K)
        x_I_prev = jnp.zeros(K)
        r_norm_sqr_prev = y_norm_sqr
        # compute the correlations of atoms with signal y
        h = Phi.T @ y
        # Pick largest 3K indices [this is first iteration]
        I_3k = largest_indices(h, K3)
        # Pick corresponding atoms to form the 3K wide subdictionary
        Phi_3I = Phi[:,I_3k]
        # Solve least squares over the selected indices
        x_3I, _, _, _ = jnp.linalg.lstsq(Phi_3I, y)
        # pick the K largest indices
        Ia = largest_indices(x_3I, K)
        # Identify indices for corresponding atoms
        I = jnp.sort(I_3k[Ia])
        # Corresponding non-zero entries in the sparse approximation
        x_I = x_3I[Ia]
        # Form the subdictionary of corresponding atoms
        Phi_I = Phi[:, I]
        # Compute new residual
        r = y - Phi_I @ x_I
        # Compute residual norm squared
        r_norm_sqr = r.T @ r
        # Assemble the algorithm state at the end of first iteration
        return CoSaMPState(x_I=x_I, I=I, r=r, r_norm_sqr=r_norm_sqr, 
            iterations=1,
            I_prev=I_prev, x_I_prev=x_I_prev, r_norm_sqr_prev=r_norm_sqr_prev)

    def body(state):
        I_prev = state.I
        x_I_prev = state.x_I
        r_norm_sqr_prev = state.r_norm_sqr
        # Index set of atoms for current solution
        I = state.I
        # compute the correlations of dictionary atoms with the residual
        h = Phi.T @ state.r
        # Ignore the previously selected atoms
        h = h.at[I].set(0)
        # Pick largest 2K indices
        I_2k = largest_indices(h, K2)
        # Combine with previous K indices to form a set of 3K indices
        I_3k = jnp.hstack((I, I_2k))
        # Pick corresponding atoms to form the 3K wide subdictionary
        Phi_3I = Phi[:, I_3k]
        # Solve least squares over the selected indices
        x_3I, r_3I_norms, rank_3I, s_3I = jnp.linalg.lstsq(Phi_3I, y)
        # pick the K largest indices
        Ia = largest_indices(x_3I, K)
        # Identify indices for corresponding atoms
        I = I_3k[Ia]
        # Corresponding non-zero entries in the sparse approximation
        x_I = x_3I[Ia]
        # Form the subdictionary of corresponding atoms
        Phi_I = Phi[:, I]
        # Compute new residual
        r = y - Phi_I @ x_I
        # Compute residual norm squared
        r_norm_sqr = r.T @ r
        return CoSaMPState(x_I=x_I, I=I, r=r, r_norm_sqr=r_norm_sqr, 
            iterations=state.iterations+1,
            I_prev=I_prev, x_I_prev=x_I_prev, r_norm_sqr_prev=r_norm_sqr_prev
            )

    def cond(state):
        # limit on residual norm 
        a = state.r_norm_sqr > max_r_norm_sqr
        # limit on number of iterations
        b = state.iterations < max_iters
        c = jnp.logical_and(a, b)
        return c

    state = lax.while_loop(cond, body, init())
    return RecoverySolution(x_I=state.x_I, I=state.I, r=state.r, r_norm_sqr=state.r_norm_sqr,
        iterations=state.iterations, length=Phi.shape[1])


matrix_solve_jit = jit(matrix_solve, static_argnums=(2,), static_argnames=("max_iters", "res_norm_rtol"))

def operator_solve(Phi, y, K, max_iters=None, res_norm_rtol=1e-4):
    r"""Solves the sparse recovery problem :math:`y = \Phi x + e` using Compressive Sampling Matching Pursuit for linear operators

    Examples:

    - :ref:`gallery:0002`
    - :ref:`gallery:0003`
    - :ref:`gallery:0004`
    - :ref:`gallery:0007`
    """
    trans = Phi.trans
    M = y.shape[0]
    ## Initialize some constants for the algorithm
    K2 = EXTRA_FACTOR * K
    K3 = K + K2
    # squared norm of the signal
    y_norm_sqr = jnp.abs(jnp.vdot(y, y))
    y_norm = jnp.sqrt(y_norm_sqr)
    # scale the signal down.
    scale = 1.0 / y_norm
    y = scale * y

    dtype = jnp.float64 if Phi.real else jnp.complex128

    max_r_norm_sqr = (res_norm_rtol ** 2)

    if max_iters is None:
        max_iters = M 

    min_iters = min(3*K, 20) 

    def init():
        # Data for the previous approximation [r = y, x = 0]
        I_prev = jnp.arange(0, K)
        x_I_prev = jnp.zeros(K, dtype=dtype)
        r_norm_sqr_prev = 1.
        # compute the correlations of atoms with signal y
        h = trans(y)
        # Pick largest 3K indices [this is first iteration]
        I_3k = largest_indices(h, K3)
        # Pick corresponding atoms to form the 3K wide subdictionary
        Phi_3I = Phi.columns(I_3k)
        # Solve least squares over the selected indices
        x_3I, r_3I_norms, rank_3I, s_3I = jnp.linalg.lstsq(Phi_3I, y)
        # pick the K largest indices
        Ia = largest_indices(x_3I, K)
        # Identify indices for corresponding atoms
        I = jnp.sort(I_3k[Ia])
        # Corresponding non-zero entries in the sparse approximation
        x_I = x_3I[Ia]
        # Form the subdictionary of corresponding atoms
        Phi_I = Phi.columns(I)
        # Compute new residual
        r = y - Phi_I @ x_I
        # Compute residual norm squared
        r_norm_sqr = jnp.abs(jnp.vdot(r, r))
        # Assemble the algorithm state at the end of first iteration
        return CoSaMPState(x_I=x_I, I=I, r=r, r_norm_sqr=r_norm_sqr, 
            iterations=1,
            I_prev=I_prev, x_I_prev=x_I_prev, r_norm_sqr_prev=r_norm_sqr_prev)

    def body(state):
        I_prev = state.I
        x_I_prev = state.x_I
        r_norm_sqr_prev = state.r_norm_sqr
        # Index set of atoms for current solution
        I = state.I
        # compute the correlations of dictionary atoms with the residual
        h = trans(state.r)
        # Ignore the previously selected atoms
        h = h.at[I].set(0)
        # Pick largest 2K indices
        I_2k = largest_indices(h, K2)
        # Combine with previous K indices to form a set of 3K indices
        I_3k = jnp.hstack((I, I_2k))
        # Pick corresponding atoms to form the 3K wide subdictionary
        Phi_3I = Phi.columns(I_3k)
        # Solve least squares over the selected indices
        x_3I, r_3I_norms, rank_3I, s_3I = jnp.linalg.lstsq(Phi_3I, y)
        # pick the K largest indices
        Ia = largest_indices(x_3I, K)
        # Identify indices for corresponding atoms
        I = I_3k[Ia]
        # Corresponding non-zero entries in the sparse approximation
        x_I = x_3I[Ia]
        # Form the subdictionary of corresponding atoms
        Phi_I = Phi.columns(I)
        # Compute new residual
        r = y - Phi_I @ x_I
        # Compute residual norm squared
        r_norm_sqr = jnp.abs(jnp.vdot(r, r))
        return CoSaMPState(x_I=x_I, I=I, r=r, r_norm_sqr=r_norm_sqr, 
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
        c = jnp.logical_and(c, d)
        # there should be some reduction in the residual norm
        # e = state.r_norm_sqr < 0.9 * state.r_norm_sqr_prev
        # c = jnp.logical_and(c, e)
        # overall condition
        return c

    state = init()
    state = lax.while_loop(cond, body, state)
    # while cond(state):
    #     state = body(state)

    # scale back the result
    x_I = y_norm * state.x_I
    r = y_norm * state.r
    r_norm_sqr = state.r_norm_sqr * y_norm_sqr
    return RecoverySolution(x_I=x_I, I=state.I, r=r, r_norm_sqr=r_norm_sqr,
        iterations=state.iterations, length=Phi.shape[1])


operator_solve_jit = jit(operator_solve, static_argnums=(0, 2), static_argnames=("max_iters", "res_norm_rtol"))

solve = operator_solve_jit
