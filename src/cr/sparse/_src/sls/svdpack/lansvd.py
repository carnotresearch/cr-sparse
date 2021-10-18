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

from typing import NamedTuple
import math

from jax import lax, jit, vmap, random
import jax.numpy as jnp
from jax.numpy.linalg import norm


import cr.sparse as crs

from .lanbpro import (LanBDOptions, LanBProState, 
    lanbpro_options_init, lanbpro_init, lanbpro_iteration, lanbpro)


class LanSVDState(NamedTuple):
    """State for Lan SVD iterations
    """
    bpro_state: LanBProState
    "State for the LanBPro algorithm"
    n_converged : int = 0
    "Number of converged eigen values"

def lansvd(A, k, p0):
    """Returns the k largest singular values and corresponding vectors
    """
    m, n = A.shape
    # maximum number of U/V columns
    lanmax = min(m, n)
    lanmax = min (lanmax, max(20, k*4))
    assert k <= lanmax, "k must be less than lanmax"

    n_converged = 0
    # number of iterations for LanBPro algorithm
    j = min(k + max(8, k), lanmax)
    options = lanbpro_options_init(lanmax)
    state  = lanbpro_init(A, lanmax, p0, options)
    while n_converged < k:
        # carry out the lanbpro iterations
        for i in range(j):
            state = lanbpro_iteration(A, state, options)
        # norm of the residual 
        res_norm = norm(state.p)
        # compute SVD of the bidiagonal matrix
        S = bdsqr(state.alpha, state.beta, res_norm)
        break
    return S



lansvd_jit = jit(lansvd, static_argnums=(0, 1))
