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
Block Sparse Bayesian Learning-Expectation Maximization


Some key assumptions in this design
- block sizes are equal and fixed
- No pruning is done
- Signal is either noisy or noiseless
"""


from jax import jit, lax
import jax.numpy as jnp


from typing import NamedTuple

import cr.nimble as crn


class BSBL_EM_Options(NamedTuple):
    """Options for the BSBL EM algorithm
    """
    learn_lambda: int = 2
    learn_type: int = 1
    prune_gamma: float = 1e-3
    lambda_val: float = 1e-14
    max_iters: int = 800
    epsilon : float = 1e-8


class BSBL_EM_State(NamedTuple):
    """Sparse Bayesian Learning algorithm state
    """
    x: jnp.ndarray
    "Solution model vector"
    active_blocks: jnp.ndarray
    "Locations of active blocks"
    gamma_est : jnp.ndarray
    "Estimated values for gamma for active blocks"
    B : jnp.ndarray
    "Estimate value of the correlation matrix"
    iterations: int
    "Number of iterations"
    lambda_est : float
    "Estimated value of the noise variance"


def bsbl_em_solve(Phi, y, blk_starts, 
    options: BSBL_EM_Options = BSBL_EM_Options()):
    
    # options
    learn_lambda = options.learn_lambda
    learn_type = options.learn_type
    prune_gamma = options.prune_gamma
    lambda_val = options.lambda_val
    max_iters = options.max_iters
    epsilon = options.epsilon

    m, n = Phi.shape
    # number of blocks
    nb = len(blk_starts)
    # identify the length of each block
    blk_lens = jnp.diff(blk_starts)
    blk_lens = jnp.append(blk_lens, n - blk_starts[-1])
    max_len = jnp.max(blk_lens)
    b_equal_blocks = crn.has_equal_values_vec(blk_lens)
    Bs = dict((i, jnp.eye(length)) for i, length in enumerate(blk_lens))
    gammas = jnp.ones(nb)
    # The blocks to keep
    keep_list = jnp.arange(nb)
    n_used = nb
    iterations = 0

    while iterations < max_iters:
        iterations += 1
        min_gammas = jnp.min(gammas)
        if min_gammas < prune_gamma:
            ...



    z = jnp.zeros(4)
    state  = BSBL_EM_State(x=z, active_blocks=z, gamma_est=z,
        B=z, iterations=0, lambda_est=0)
    return z


