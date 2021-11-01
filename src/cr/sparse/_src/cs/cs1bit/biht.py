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

from typing import NamedTuple, List, Dict


import jax.numpy as jnp
from jax import vmap, jit, lax
from jax.numpy.linalg import norm

from cr.sparse import hard_threshold, build_signal_from_indices_and_values


class BIHTState(NamedTuple):
    # The non-zero values
    x_I: jnp.ndarray
    """Non-zero values"""
    I: jnp.ndarray
    """The support for non-zero values"""
    r: jnp.ndarray
    """The residuals"""
    n_mismatched_bits: jnp.ndarray
    # """The number of bits that are mismatched"""
    iterations: int
    """The number of iterations it took to complete"""


def biht(Phi, y, K, tau, max_iters=1000):
    """Solves the 1-bit compressive sensing problem :math:`y = \text{sgn} (\\Phi x)` using 
    Binary Iterative Hard Thresholding
    """
    ## Initialize some constants for the algorithm
    M, N = Phi.shape

    def init():
        # Data for the initial approximation [r = y, x = 0]
        I = jnp.arange(0, K)
        x_I = jnp.zeros(K)
        # Assume initial estimate to be zero and compute residual
        # compute the 1 bit output based on current x estimate
        y_int = Phi[:,I] @ x_I
        y_hat = jnp.sign(y_int)
        # difference between actual y and current estimate
        r = y - y_hat
        n_mismatched_bits = jnp.sum(r != 0)
        return BIHTState(x_I=x_I, I=I, r=r, n_mismatched_bits=n_mismatched_bits, 
            iterations=0)

    def body(state):
        # Compute the correlation of the residual with the atoms of Phi
        h = Phi.T @ state.r
        # current approximation
        x = build_signal_from_indices_and_values(N, state.I, state.x_I)
        # update
        x = x + tau/2 * h
        # threshold
        I, x_I = hard_threshold(x, K)
        # Form the subdictionary of corresponding atoms
        Phi_I = Phi[:, I]
        # compute the 1 bit output based on current x estimate
        y_int = Phi_I @ x_I
        y_hat = jnp.sign(y_int)
        # Compute new residual
        # difference between actual y and current estimate
        r = y - y_hat
        # Compute residual norm squared
        n_mismatched_bits = jnp.sum(r != 0)
        # update new state
        return BIHTState(x_I=x_I, I=I, r=r, n_mismatched_bits=n_mismatched_bits,
            iterations=state.iterations+1)

    def cond(state):
        # limit on residual norm 
        a = state.n_mismatched_bits > 0
        # limit on number of iterations
        b = state.iterations < max_iters
        c = jnp.logical_and(a, b)
        # overall condition
        return c

    state = lax.while_loop(cond, body, init())
    return state


biht_jit = jit(biht, static_argnums=(2))
