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

from typing import NamedTuple, List, Dict

import jax.numpy as jnp
from jax import vmap, jit, lax

import cr.nimble as crn
from .util import abs_max_idx, gram_chol_update
from cr.nimble import solve_spd_chol
from .defs import RecoverySolution

def matrix_solve(Phi, y, max_iters, max_res_norm=1e-6):
    """Solves the recovery/approximation problem :math:`y = \\Phi x + e` using Orthogonal Matching Pursuit

    Args:
        Phi: A matrix representing the underdetermined linear system
        y (jax.numpy.ndarray): The signal to be modeled by OMP
        max_iters (int): Sparsity of the solution vector (number of nonzero entries)

    Note:

        In order to support JIT compilation of this function, the halting
        criteria for residual norm has been ignored for now.
        The current implementation simply unrolls OMP main loop
        depending on max_iters. This may be revised in future.
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
    return RecoverySolution(x_I=x_I, I=I, r=r, r_norm_sqr=norm_r_new_sqr, iterations=k+1, length=Phi.shape[1])



matrix_solve_jit = jit(matrix_solve, static_argnums=(2,), static_argnames=("max_res_norm",))

matrix_solve_multi = vmap(matrix_solve_jit, (None, 1, None), 0)
"""Solves the MMV recovery/approximation problem :math:`Y = \\Phi X + E` using Orthogonal Matching Pursuit

Extends :py:func:`cr.sparse.pursuit.omp.solve` using :py:func:`jax.vmap`.
"""
solve = matrix_solve_jit


######################################################################################
#  OMP implementation for linear operators
######################################################################################


class OMPState(NamedTuple):
    # The non-zero values
    x_I: jnp.ndarray
    """Non-zero values"""
    I: jnp.ndarray
    """The support for non-zero values"""
    r: jnp.ndarray
    """The residuals"""
    Phi_I: jnp.ndarray
    "Part of the dictionary containing the chosen atoms"
    L : jnp.ndarray
    "Part of the cholesky decomposition being maintained"
    r_norm_sqr: jnp.ndarray
    "The residual norm squared"
    iterations: int
    "The number of iterations it took to complete"

    def __str__(self):
        """Returns the string representation
        """
        s = []
        for x in [
            f"iterations {self.iterations}",
            f"r_norm_sqr {self.r_norm_sqr:.2e}",
            ]:
            s.append(x.rstrip())
        return u'\n'.join(s)

def _operator_step(times, trans, y, p, zv, state):
    # iteration number
    k = state.iterations
    # compute the correlations
    h = trans(state.r)
    # Index of best match
    i = abs_max_idx(h)
    # Update the set of indices
    I = jnp.append(state.I, i)
    # best matching atom
    phi_i = times(zv.at[i].set(1))
    # Correlate with previously selected atoms
    Phi_I = state.Phi_I
    v = crn.AH_v(Phi_I, phi_i)
    # Update the Cholesky factorization
    L = gram_chol_update(state.L, v)
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

    return OMPState(x_I=x_I, I=I, Phi_I=Phi_I, r=r, L=L, 
        r_norm_sqr=norm_r_new_sqr, iterations=k + 1)


_operator_step_jit = jit(_operator_step, static_argnums=(0,1))

def operator_solve(Phi, y, K, res_norm_rtol=1e-4):
    r"""Solves the sparse recovery problem :math:`y = \Phi x + e` using Orthogonal Matching Pursuit for linear operators


    Args:
        Phi: A linear operator representing the underdetermined linear system
        y (jax.numpy.ndarray): The signal to be modeled by OMP
        K (int): Sparsity of the solution vector (number of nonzero entries)
        res_norm_rtol (float): Relative tolerance for residual norm (halting criteria)


    Note:

        * This function cannot be JIT compiled. However the main body of the loop has been JIT compiled
          for performance.
    """
    trans = Phi.trans
    times = Phi.times
    m, n = Phi.shape
    # squared norm of the signal
    y_norm_sqr = jnp.abs(jnp.vdot(y, y))
    y_norm = jnp.sqrt(y_norm_sqr)
    # scale the signal down.
    scale = 1.0 / y_norm
    y = scale * y

    dtype = jnp.float64 if Phi.real else jnp.complex128
    max_r_norm_sqr = (res_norm_rtol ** 2) 

    # an all zeros vector
    zv = jnp.zeros(n)

    # The proxy representation
    p = trans(y)


    @jit
    def init():
        # We need to carry out one iteration to initialize
        # the variables properly
        # Index of best match
        i = abs_max_idx(p)
        # Add to the array of selected indices
        I = jnp.array([i])
        # First matched atom
        phi_0 = times(zv.at[i].set(1))
        # Initial subdictionary of selected atoms
        Phi_I = jnp.expand_dims(phi_0, axis=1)
        # Initial L for Cholesky factorization of Gram matrix
        L = jnp.ones((1,1))
        # Initial coefficient
        x0 = p[i]
        x_I = x0
        # updated residual after first iteration
        r = y - x0 * phi_0
        # norm squared of new residual
        norm_r_new_sqr = jnp.abs(jnp.vdot(y, y))
        return OMPState(x_I=x_I, I=I, r=r, 
            Phi_I=Phi_I, L=L, 
            r_norm_sqr=norm_r_new_sqr, iterations=1)


    def body_func(state):
        return _operator_step_jit(times, trans, y, p, zv, state)

    def cond_func(state):
        # limit on residual norm 
        a = state.r_norm_sqr > max_r_norm_sqr
        # limit on number of iterations
        b = state.iterations < K
        c = jnp.logical_and(a, b)
        return c

    state = init()
    while(cond_func(state)):
        state = body_func(state)
    # scale back the result
    x_I = y_norm * state.x_I
    r = y_norm * state.r
    r_norm_sqr = state.r_norm_sqr * y_norm_sqr
    return RecoverySolution(x_I=x_I, I=state.I, r=r, 
        r_norm_sqr=r_norm_sqr,
        iterations=state.iterations, length=n)

