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

from typing import NamedTuple
import math

from jax import lax, jit, vmap, random
import jax.numpy as jnp
from jax.numpy.linalg import norm


import cr.nimble as cnb

from .lanbpro import (LanBDOptions, LanBProState, 
    lanbpro_options_init, lanbpro_init, lanbpro_iteration, lanbpro)

from cr.nimble.svd import bdsqr, refine_bounds

class LanSVDState(NamedTuple):
    """State for Lan SVD iterations
    """
    bpro_state: LanBProState
    "State for the LanBPro algorithm"
    n_converged : int = 0
    "Number of converged eigen values"

def lansvd_simple(A, k, p0):
    """Returns the k largest singular values and corresponding vectors
    """
    m, n = A.shape
    # maximum number of U/V columns
    lanmax = min(m, n)
    lanmax = min (lanmax, max(20, k*4))
    assert k <= lanmax, "k must be less than lanmax"

    eps = jnp.finfo(float).eps
    tol = 16*eps
    n_converged = 0
    # number of iterations for LanBPro algorithm
    j = min(k + max(8, k), lanmax)
    options = lanbpro_options_init(lanmax)
    state  = lanbpro_init(A, lanmax, p0, options)
    # carry out the lanbpro iterations
    state = lax.fori_loop(1, j, 
        lambda i, state: lanbpro_iteration(A, state, options),
        state)
    # norm of the residual 
    res_norm = norm(state.p)
    # compute SVD of the bidiagonal matrix ritz values and vectors
    P, S, Qh, bot = bdsqr(state.alpha, state.beta, j)
    # estimate of A norm 
    anorm = S[0]
    # simple error bounds on singular values
    bnd = res_norm * jnp.abs(bot)
    # now refine the bounds
    bnd = refine_bounds(S**2, bnd, n*eps*anorm)
    # count the number of converged singular values
    converged = jnp.less(bnd, jnp.abs(S))
    # make sure that all indices beyond min(j,k) are marked as non-converged
    converged = converged.at[min(j,k):].set(False)
    # find the index of first non-converged singular value
    n_converged = jnp.argmin(converged)
    n_converged = jnp.where(converged[n_converged], len(converged), n_converged)
    U = state.U[:, :j]
    V = state.V[:, :j]
    # keep only the first k ritz vectors
    P = P[:, :k]
    Q = Qh.T[:, :k]
    U = state.U[:, :j] @ P[:j, :]
    V = state.V[:, :j] @ Q
    return U, S[:k], V, bnd, n_converged, state



lansvd_simple_jit = jit(lansvd_simple, static_argnums=(0, 1))
