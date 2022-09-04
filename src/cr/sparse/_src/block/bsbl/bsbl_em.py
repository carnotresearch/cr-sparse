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

import math

from jax import jit, lax
import jax.numpy as jnp
norm = jnp.linalg.norm

from typing import NamedTuple

import cr.nimble as crn
import cr.sparse.plots as crplot
from .bsbl import *


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
    mu_x: jnp.ndarray
    "Mean vectors for each block"
    r: jnp.ndarray
    "The residuals"
    r_norm_sqr: jnp.ndarray
    "The residual norm squared"
    gammas : jnp.ndarray
    "Estimated values for gamma for each block"
    Sigma0: jnp.ndarray
    "Prior correlation matrices for each block"    
    lambda_val : float
    "Estimated value of the noise variance"
    dmu: float
    "Maximum absolute difference between two iterations for means"
    iterations: int
    "Number of iterations"


    @property
    def x(self):
        return self.mu_x.flatten()

    def __str__(self):
        """Returns the string representation
        """
        s = []
        r_norm = math.sqrt(float(self.r_norm_sqr))
        x_norm = float(norm(self.x))
        n_blocks, blk_size, _ = self.Sigma0.shape
        for x in [
            u"iterations %s" % self.iterations,
            f"blocks={n_blocks}, block size={blk_size}",
            u"r_norm %e" % r_norm,
            u"x_norm %e" % x_norm,
            u"lambda %e" % self.lambda_val,
            u"dmu %e" % float(self.dmu),
            ]:
            s.append(x.rstrip())
        return u'\n'.join(s)


def bsbl_em(Phi, y, blk_len, 
    options: BSBL_EM_Options = BSBL_EM_Options()):
    
    # options
    learn_lambda = options.learn_lambda
    learn_type = options.learn_type
    prune_gamma = options.prune_gamma
    lambda_val = options.lambda_val
    max_iters = options.max_iters
    epsilon = options.epsilon
    # measurement and model space dimensions
    m, n = Phi.shape
    # length of each block
    b = blk_len
    # number of blocks
    nb = n // b
    # split Phi into blocks
    Subdicts = get_subdicts(Phi, nb)

    # y scaling
    y_norm_sqr = crn.sqr_norm_l2(y)

    # start solving

    def init_func():
        # initialize posterior means for each block
        mu_x = jnp.zeros((nb, b))
        # initialize correlation matrices
        Sigma0 = init_sigmas(n, b)
        # initialize block correlation scalars
        gammas = init_gammas(nb)
        state = BSBL_EM_State(
            mu_x=mu_x,
            r=y,
            r_norm_sqr=y_norm_sqr,
            gammas=gammas,
            Sigma0=Sigma0,
            lambda_val=lambda_val,
            dmu=1.,
            iterations=0)
        return state

    def body_func(state):
        # active_blocks = gammas > prune_gamma
        PhiBPhi = cum_phi_b_phi(Subdicts, state.Sigma0)
        H = compute_h(Phi, PhiBPhi, state.lambda_val)
        # posterior block means
        mu_x = compute_mu_x(state.Sigma0, H, y)
        # posterior block covariances
        Sigma_x = compute_sigma_x(Phi, state.Sigma0, H)
        Cov_x = compute_cov_x(Sigma_x, mu_x)
        Bi_sum = compute_cov_x_sum(Cov_x, state.gammas)
        B, B_inv = compute_B_B_inv(Bi_sum)
        # flattened signal
        x_hat = mu_x.flatten()
        # residual
        res = y - Phi @ x_hat
        # residual norm squared
        r_norm_sqr = crn.sqr_norm_l2(res)
        # print(f'r_norm_sqr: {r_norm_sqr:e}')
        # update lambda
        lambda_val = state.lambda_val
        # update gamma
        gammas = update_gammas(Cov_x, B_inv)
        # update sigma
        Sigma0 = update_sigma_0(gammas, B)

        # convergence criterion
        mu_diff = jnp.abs(mu_x - state.mu_x)
        dmu = jnp.max(mu_diff)

        state = BSBL_EM_State(
            mu_x=mu_x,
            r=res,
            r_norm_sqr=r_norm_sqr,
            gammas=gammas,
            Sigma0=Sigma0,
            lambda_val=lambda_val,
            dmu=dmu,
            iterations=state.iterations + 1)
        return state


    def cond_func(state):
        a = state.dmu > epsilon
        b = state.iterations < max_iters
        c = jnp.logical_and(a, b)
        return c

    state = lax.while_loop(cond_func, body_func, init_func())
    # state = init_func()
    # while cond_func(state):
    #     state = body_func(state)
    return state


bsbl_em_jit = jit(bsbl_em, static_argnums=(2,))