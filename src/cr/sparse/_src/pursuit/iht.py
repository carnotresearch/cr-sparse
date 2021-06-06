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

from cr.sparse import hard_threshold, build_signal_from_indices_and_values
from cr.sparse.dict import upper_frame_bound

def solve(Phi, y, K, step_size=None, max_iters=None, res_norm_rtol=1e-3):
    """Solves the sparse recovery problem :math:`y = \Phi x + e` using Iterative Hard Thresholding
    """
    ## Initialize some constants for the algorithm
    M, N = Phi.shape

    # squared norm of the signal
    y_norm_sqr = y.T @ y

    max_r_norm_sqr = y_norm_sqr * (res_norm_rtol ** 2)

    if step_size is None:
        step_size = 0.98 / upper_frame_bound(Phi)

    if max_iters is None:
        max_iters = M**2 

    def init():
        # Assume previous estimate to be zero and conduct first iteration
        # compute the correlations of atoms with signal y
        h = Phi.T @ y
        # update
        x = step_size * h
        # threshold
        I, x_I = hard_threshold(x, K)
        # Form the subdictionary of corresponding atoms
        Phi_I = Phi[:, I]
        # Compute new residual
        r = y - Phi_I @ x_I
        # Compute residual norm squared
        r_norm_sqr = r.T @ r
        return RecoverySolution(x_I=x_I, I=I, r=r, r_norm_sqr=r_norm_sqr, iterations=1)

    def iteration(state):
        # compute the correlations of dictionary atoms with the residual
        h = Phi.T @ state.r
        # current approximation
        x = build_signal_from_indices_and_values(N, state.I, state.x_I)
        # update
        x = x + step_size * h
        # threshold
        I, x_I = hard_threshold(x, K)
        # Form the subdictionary of corresponding atoms
        Phi_I = Phi[:, I]
        # Compute new residual
        r = y - Phi_I @ x_I
        # Compute residual norm squared
        r_norm_sqr = r.T @ r
        return RecoverySolution(x_I=x_I, I=I, r=r, r_norm_sqr=r_norm_sqr, iterations=state.iterations+1)

    def cond(state):
        # limit on residual norm and number of iterations
        return jnp.logical_and(state.r_norm_sqr > max_r_norm_sqr, state.iterations < max_iters)

    state = lax.while_loop(cond, iteration, init())
    return state
