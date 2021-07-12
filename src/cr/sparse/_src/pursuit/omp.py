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

from .util import abs_max_idx, gram_chol_update
from cr.sparse.la import solve_spd_chol
from .defs import RecoverySolution

def matrix_solve(Phi, y, max_iters, max_res_norm=1e-6):
    """Solves the recovery/approximation problem :math:`y = \Phi x + e` using Orthogonal Matching Pursuit
    """
    # initialize residual
    r = y
    D = Phi.shape[0]
    N = Phi.shape[1]
    K = max_iters
    # Let's conduct first iteration of OMP
    # squared norm of the signal
    norm_y_sqr = y.T @ y
    # initialize residual squared norm with the same
    norm_r_sqr = norm_y_sqr
    # The proxy representation
    p = Phi.T @ y
    # First correlation of residual with signal
    h = p
    # Index of best match
    i = abs_max_idx(h)
    # Initialize the array of selected indices
    I = jnp.array([i])
    # First matched atom
    phi_i = Phi[:, i]
    # Initial subdictionary of selected atoms
    Phi_I = jnp.expand_dims(phi_i, axis=1)
    # Initial L for Cholesky factorization of Gram matrix
    L = jnp.ones((1,1))
    # sub-vector of proxy corresponding to selected indices
    p_I = p[I]
    # sub-vector of representation coefficients estimated so far
    x_I = p_I
    # updated residual after first iteration
    r = y - Phi_I @ x_I
    # norm squared of new residual
    norm_r_new_sqr = r.T @ r
    # conduct OMP iterations
    for k in range(1, K):
        norm_r_sqr = norm_r_new_sqr
        # compute the correlations
        h = Phi.T @ r
        # Index of best match
        i = abs_max_idx(h)
        # Update the set of indices
        I = jnp.append(I, i)
        # best matching atom
        phi_i = Phi[:, i]
        # Correlate with previously selected atoms
        v = Phi_I.T @ phi_i
        # Update the Cholesky factorization
        L = gram_chol_update(L, v)
        # Update the subdictionary
        Phi_I = jnp.hstack((Phi_I, jnp.expand_dims(phi_i,1)))
        # sub-vector of proxy corresponding to selected indices
        p_I = p[I]
        # sub-vector of representation coefficients estimated so far
        x_I = solve_spd_chol(L, p_I)
        # updated residual after first iteration
        r = y - Phi_I @ x_I
        # norm squared of new residual
        norm_r_new_sqr = r.T @ r
    return RecoverySolution(x_I=x_I, I=I, r=r, r_norm_sqr=norm_r_new_sqr, iterations=k+1)



matrix_solve_jit = jit(matrix_solve, static_argnums=(2,), static_argnames=("max_res_norm",))

matrix_solve_multi = vmap(matrix_solve_jit, (None, 1, None), 0)
"""Solves the MMV recovery/approximation problem :math:`Y = \Phi X + E` using Orthogonal Matching Pursuit

Extends :py:func:`cr.sparse.pursuit.omp.solve` using :py:func:`jax.vmap`.
"""
solve = matrix_solve_jit
