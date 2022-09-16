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
Block Sparse Bayesian Learning

* Expectation Maximization
* Bound Optimization


Some assumptions in this design

* block sizes are equal and user defined
"""

import math
from typing import NamedTuple

from jax import jit, vmap, lax
from jax.lax import fori_loop
import jax.numpy as jnp
norm = jnp.linalg.norm
import jax.scipy.linalg
sqrtm = jax.scipy.linalg.sqrtm

import cr.nimble as crn
import cr.sparse.block.block as crblock
import cr.sparse.plots as crplot


def sqrtm_svd(B):
    """Square root of a symmetric matrix using its SVD
    """
    u, s, vh = jnp.linalg.svd(B)
    s = jnp.sqrt(s)
    return u @ jnp.diag(s) @ vh

def init_sigmas(n, b):
    n_blocks = n // b
    I = jnp.eye(b)
    return jnp.broadcast_to(I, (n_blocks,) + I.shape)


def init_gammas(n_blocks):
    return jnp.ones(n_blocks)

def prune_blocks(gammas, threshold):
    return gammas > threshold

def get_subdicts(Phi, n_blocks):
    m, n = Phi.shape
    subdicts = Phi.swapaxes(0, 1).reshape(n_blocks, -1, m).swapaxes(1,2)
    return subdicts

def phi_b_phi(Phi, start, length, Sigma0):
    subdict = Phi[:, start: start + length]
    return subdict @ Sigma0 @ subdict.T

def cum_phi_b_phi_ref(Phi, Sigma0):
    n_blocks = len(Sigma0)
    m, n = Phi.shape
    # block length
    b = n // n_blocks
    starts = [i*b for i in range(n_blocks)]
    result = jnp.zeros((m, m))
    # zero value
    z = result
    for i in range(n_blocks):
        result += phi_b_phi(Phi, starts[i], b, Sigma0[i])
    return result



def cum_phi_b_phi(Subdicts, Sigma0):
    phi_b_phis = vmap(
        lambda subdict, s:  subdict @ s @ subdict.T,
        in_axes=(0, 0))(Subdicts, Sigma0)
    return jnp.sum(phi_b_phis, axis=0)

def cum_phi_b_phi_pruned(Subdicts, Sigma0, active_blocks):
    m = Subdicts.shape[1]
    result = jnp.zeros((m, m))
    # zero value
    z = result
    phi_b_phis = vmap(
        lambda subdict, s, active:  lax.cond(active,
            lambda _: subdict @ s @ subdict.T,
            lambda _: z,
            None),
        in_axes=(0, 0, 0))(Subdicts, Sigma0, active_blocks)
    return jnp.sum(phi_b_phis, axis=0)

def compute_h(Phi, PhiBPhi, lambda_val):
    n = PhiBPhi.shape[0]
    # PhiBPhi + lambda I
    A = crn.add_to_diagonal(PhiBPhi, lambda_val)
    HT = jnp.linalg.solve(A, Phi)
    return HT.T


def compute_mu_x(Sigma0, H, y):
    Hy = H @ y
    n_blocks = len(Sigma0)
    # split Hy into blocks
    Hy = jnp.reshape(Hy, (n_blocks, -1))
    # compute x means
    mu_x = vmap(lambda a, y: a @ y, in_axes=(0, 0))(Sigma0, Hy)
    return mu_x, Hy


def compute_mu_x_pruned(Sigma0, H, y, active_blocks):
    Hy = H @ y
    n_blocks, blk_size, _ = Sigma0.shape
    # split Hy into blocks
    Hy = jnp.reshape(Hy, (n_blocks, -1))
    # zero mean for inactive blocks
    z = jnp.zeros(blk_size)
    # compute x means
    mu_x = vmap(
        lambda a, y, active: lax.cond(active,
            lambda _ : a @ y,
            lambda _ : z,
            None), 
        in_axes=(0, 0, 0)
        )(Sigma0, Hy, active_blocks)
    return mu_x, Hy


def compute_sigma_x(Phi, Sigma0, H):
    # block length
    b = Sigma0.shape[1]
    HPhi = H @ Phi
    # Extract the block diagonals
    HPhi_blocks = crn.block_diag(HPhi, b)
    Sigma_x = vmap(
        lambda A, B: A - A  @ B @ A,
        in_axes=(0, 0))(Sigma0, HPhi_blocks)
    return Sigma_x, HPhi_blocks

def compute_sigma_x_pruned(Phi, Sigma0, H, active_blocks):
    n_blocks, blk_size, _ = Sigma0.shape
    HPhi = H @ Phi
    # Extract the block diagonals
    HPhi_blocks = crn.block_diag(HPhi, blk_size)
    # zero valued blocks
    z = jnp.zeros((blk_size, blk_size))
    Sigma_x = vmap(
        lambda A, B, active: lax.cond(active,
            lambda _ : A - A  @ B @ A,
            lambda _ : z,
            None),
        in_axes=(0, 0, 0))(Sigma0, HPhi_blocks, active_blocks)
    return Sigma_x, HPhi_blocks

def compute_cov_x(Sigma_x, mu_x):
    Cov_x = vmap(
        lambda sx, mx: sx + mx @ mx.T,
        in_axes=(0,0))(Sigma_x, mu_x)
    return Cov_x


def compute_cov_x_sum(Cov_x, gammas):
    B_i = vmap(
        lambda cx, g: cx / g,
        in_axes=(0, 0))(Cov_x, gammas)
    return jnp.sum(B_i, axis=0)


def compute_cov_x_sum_pruned(Cov_x, gammas, active_blocks):
    B_i = vmap(
        lambda cx, g, active: lax.cond(active,
            lambda _: cx / g,
            lambda _ : cx,
            None),
        in_axes=(0, 0, 0))(Cov_x, gammas, active_blocks)
    return jnp.sum(B_i, axis=0)

def compute_B_B_inv(B0):
    m0 = jnp.mean(jnp.diag(B0))
    m1 = jnp.mean(jnp.diag(B0, 1))
    # AR-1 coefficient
    r = m1 / m0
    # make sure that it is bounded
    r = jnp.clip(r, -0.99, 0.99)
    # print(f'r: {r}')
    # block size
    b = B0.shape[0]
    c = r ** jnp.arange(b)
    B = crn.toeplitz_mat(c, c)
    B_inv = jnp.linalg.inv(B)
    return B, B_inv

# This rule doesn't seem to work in noiseless case
def update_lambda_high_snr(lambda_val, gammas, Sigma_x, B_inv, r_norm_sqr, m):
    n_blocks = len(gammas)
    ll = vmap(
        lambda g, sx: jnp.trace(sx @ B_inv) / g,
        in_axes=(0, 0))(gammas, Sigma_x)
    lambda_comp = jnp.sum(ll)
    carry = lambda_val * (n_blocks - lambda_comp)
    new_lambda_val = (r_norm_sqr + carry) / m
    # print(f'old:{lambda_val:.2e}, new:{new_lambda_val:.2e}, comp: {lambda_comp:.2e}, carry: {carry:.2e}, r_norm_sqr: {r_norm_sqr:.2e}')
    return new_lambda_val

def update_lambda_low_snr(lambda_val, Subdicts, Sigma_x, r_norm_sqr, m):
    ll = vmap(
        lambda subdict, s:  jnp.trace(subdict @ s @ subdict.T),
        in_axes=(0, 0))(Subdicts, Sigma_x)
    lambda_comp = jnp.sum(ll)
    new_lambda_val = (r_norm_sqr + lambda_comp) / m
    # print(f'old:{lambda_val:.2e}, new:{new_lambda_val:.2e}, comp: {lambda_comp:.2e}, r_norm_sqr: {r_norm_sqr:.2e}')
    return new_lambda_val

def update_lambda_rule_nojit(learn_lambda, 
    lambda_val, Subdicts, gammas, Sigma_x, B_inv, r_norm_sqr, m):
    if learn_lambda == 0:
        return lambda_val
    if learn_lambda == 1:
        return update_lambda_low_snr(lambda_val, Subdicts, Sigma_x, r_norm_sqr, m)
    return update_lambda_high_snr(lambda_val, gammas, Sigma_x, B_inv, r_norm_sqr, m)


def update_lambda_rule_jittable(learn_lambda, 
    lambda_val, Subdicts, gammas, Sigma_x, B_inv, r_norm_sqr, m):
    return lax.switch(learn_lambda,
        [
        lambda lambda_val: lambda_val,
        lambda lambda_val: update_lambda_low_snr(
            lambda_val, Subdicts, Sigma_x, r_norm_sqr, m),
        lambda lambda_val: update_lambda_high_snr(lambda_val, 
        gammas, Sigma_x,
        B_inv, r_norm_sqr, m),
        ], lambda_val)

def update_gammas_em(Cov_x, B_inv):
    n_blocks, blk_size, _ = Cov_x.shape
    gammas = vmap(
        lambda cx: jnp.trace(B_inv @ cx))(Cov_x)
    gammas = gammas / blk_size
    return gammas


def update_gammas_em_pruned(Cov_x, B_inv, active_blocks):
    n_blocks, blk_size, _ = Cov_x.shape
    gammas = vmap(
        lambda cx, active: lax.cond(active,
            lambda _: jnp.trace(B_inv @ cx),
            lambda _: 0., 
            None),
        in_axes=(0, 0)
        )(Cov_x, active_blocks)
    gammas = gammas / blk_size
    return gammas

def update_gammas_bo(old_gammas, B, Hy, HPhi):
    n_blocks = len(old_gammas)
    blk_size = B.shape[0]
    B_root = sqrtm_svd(B)

    def mapper(g, hy, hphi):
        numer = norm(B_root @ hy)
        denom = jnp.sqrt(jnp.trace(hphi @ B))
        result =  g * numer / denom
        return result

    gammas = vmap(mapper, in_axes=(0, 0, 0))(old_gammas, Hy, HPhi)
    return gammas

def update_gammas_bo_pruned(old_gammas, B, Hy, HPhi, active_blocks):
    n_blocks = len(old_gammas)
    blk_size = B.shape[0]
    B_root = sqrtm_svd(B)

    def mapper(g, hy, hphi):
        numer = norm(B_root @ hy)
        denom = jnp.sqrt(jnp.trace(hphi @ B))
        result =  g * numer / denom
        return result

    gammas = vmap(
        lambda g, hy, hphi, active: lax.cond(active,
            lambda _: mapper(g, hy, hphi),
            lambda _: 0.,
            None), 
        in_axes=(0, 0, 0, 0))(old_gammas, Hy, HPhi, active_blocks)
    return gammas


def update_sigma_0(gammas, B):
    Sigma0 = vmap(
        lambda gamma: gamma * B,
        in_axes=(0,))(gammas)
    return Sigma0


##################################################
# Options for BSBL Algorithm
##################################################

class BSBL_Options(NamedTuple):
    """Options for the BSBL algorithm
    """
    learn_block_corr: int = 1
    """Indicates whether intra block correlations are to
    be learned or ignored.

    * 0: Ignore intra block correlation
    * 1: Learn intra block correlation
    """
    learn_lambda: int = 1
    """Indicates if the variance of noise is to be estimated from data

    * 0: Do not learn lambda. Use the input or default value.
    * 1: Use the lambda learning rule for noisy cases [SNR <= 20 dB]
    * 2: Use the lambda learning rule for high SNR cases[SNR > 20 dB]
    """
    prune_gamma: float = 1e-3
    """Threshold for pruning small values of gamma_i
    """
    lambda_val: float = 1e-12
    """User defined value for lambda [if provided by user]
    """
    max_iters: int = 800
    """Maximum number of iterations for the BSBL algorithm
    """
    epsilon : float = 1e-8
    """Solution accuracy tolerance parameter
    """



def bsbl_em_options(y=None, 
    learn_block_corr=None,
    learn_lambda=None,
    prune_gamma=None,
    lambda_val=None, max_iters=None,
    epsilon=None):
    """Helper function to initialize options for the BSBL-EM algorithm
    """
    # default values of options
    opt  = BSBL_Options()
    # customize them
    learn_block_corr = opt.learn_block_corr if learn_block_corr is None else learn_block_corr
    learn_lambda = opt.learn_lambda if learn_lambda is None else learn_lambda
    epsilon = opt.epsilon if epsilon is None else epsilon
    max_iters = opt.max_iters if max_iters is None else max_iters

    if learn_lambda == 0:
        # Noise-less
        lambda_val_ = 1e-12
        prune_gamma_ = 1e-3
    elif learn_lambda == 1:
        # Low SNR
        lambda_val_ = 1e-3
        prune_gamma_ = 1e-2
    else:
        # High SNR
        lambda_val_ = 1e-3
        prune_gamma_ = 1e-2
    prune_gamma = prune_gamma_ if prune_gamma is None else prune_gamma
    lambda_val = lambda_val_ if lambda_val is None else lambda_val

    return BSBL_Options(learn_block_corr=learn_block_corr,
        learn_lambda=learn_lambda,
        prune_gamma=prune_gamma,
        lambda_val=lambda_val,
        max_iters=max_iters,
        epsilon=epsilon)


def bsbl_bo_options(y=None, 
    learn_block_corr=None,
    learn_lambda=None,
    prune_gamma=None,
    lambda_val=None, max_iters=None,
    epsilon=None):
    """Helper function to initialize options for the BSBL-BO algorithm
    """
    scale = jnp.std(y) if y is not None else 1.
    # default values of options
    opt  = BSBL_Options()
    # customize them
    learn_block_corr = opt.learn_block_corr if learn_block_corr is None else learn_block_corr
    learn_lambda = opt.learn_lambda if learn_lambda is None else learn_lambda
    epsilon = opt.epsilon if epsilon is None else epsilon
    max_iters = 300 if max_iters is None else max_iters

    if learn_lambda == 0:
        # Noise-less
        lambda_val_ = 1e-12
        prune_gamma_ = 1e-3
    elif learn_lambda == 1:
        # Low SNR
        lambda_val_ = scale * 1e-2
        prune_gamma_ = 1e-2
    else:
        # High SNR
        lambda_val_ = scale * 1e-2
        prune_gamma_ = 1e-2
    prune_gamma = prune_gamma_ if prune_gamma is None else prune_gamma
    lambda_val = lambda_val_ if lambda_val is None else lambda_val

    return BSBL_Options(learn_block_corr=learn_block_corr,
        learn_lambda=learn_lambda,
        prune_gamma=prune_gamma,
        lambda_val=lambda_val,
        max_iters=max_iters,
        epsilon=epsilon)


##################################################
# BSBL Algorithm State
##################################################

class BSBL_State(NamedTuple):
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
        "Solution vector"
        return self.mu_x.flatten()

    def __str__(self):
        """Returns the string representation
        """
        s = []
        r_norm = math.sqrt(float(self.r_norm_sqr))
        x_norm = float(norm(self.x))
        n_blocks, blk_size, _ = self.Sigma0.shape
        n_active = jnp.sum(self.gammas > 0)
        for x in [
            f"iterations={self.iterations}",
            f'block size={blk_size}',
            f"blocks={n_blocks}, nonzero={n_active}",
            u"r_norm=%.2e" % r_norm,
            u"x_norm=%.2e" % x_norm,
            u"lambda=%.2e" % self.lambda_val,
            u"dmu=%.2e" % float(self.dmu),
            ]:
            s.append(x.rstrip())
        return u'\n'.join(s)


##################################################
# BSBL Expectation Maximization
##################################################

def bsbl_em(Phi, y, blk_len, 
    options: BSBL_Options = BSBL_Options()):
    """Reconstructs a block sparse signal using BSBL-EM algorithm

    Args:
        Phi (jax.numpy.ndarray): Sensing matrix
        y (jax.numpy.ndarray): Measurement vector
        blk_len (int): Length/size of each block
        options (BSBL_Options): Options for algorithm execution

    Returns:
        BSBL_State: Solution of the sparse recovery problem

    Note:
        * Phi must be a matrix. Linear operators are not supported
          since we need to break Phi down into submatrices for each block.
        * Use :py:func:`bsbl_em_options` to initialize options for
          the algorithm.

    Examples:
        - :ref:`gallery:cs:bsbl:1`
    """
    
    # options
    learn_lambda = options.learn_lambda
    learn_block_corr = options.learn_block_corr
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
        state = BSBL_State(
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
        active_blocks = state.gammas > prune_gamma
        PhiBPhi = cum_phi_b_phi_pruned(Subdicts, state.Sigma0, active_blocks)
        H = compute_h(Phi, PhiBPhi, state.lambda_val)
        # posterior block means
        mu_x, _ = compute_mu_x_pruned(state.Sigma0, H, y, active_blocks)
        # posterior block covariances
        Sigma_x, _ = compute_sigma_x_pruned(Phi, state.Sigma0, H, active_blocks)
        Cov_x = compute_cov_x(Sigma_x, mu_x)
        Bi_sum = compute_cov_x_sum_pruned(Cov_x, state.gammas, active_blocks)
        B, B_inv = compute_B_B_inv(Bi_sum)
        # flattened signal
        x_hat = mu_x.flatten()
        # residual
        res = y - Phi @ x_hat
        # residual norm squared
        r_norm_sqr = crn.sqr_norm_l2(res)
        # update lambda
        # lambda_val = update_lambda_rule_nojit(learn_lambda,
        #     state.lambda_val, Subdicts, state.gammas, Sigma_x,
        #     B_inv, r_norm_sqr, m)
        lambda_val = update_lambda_rule_jittable(learn_lambda,
            state.lambda_val, Subdicts, state.gammas, Sigma_x,
            B_inv, r_norm_sqr, m)
        # update gamma
        gammas = update_gammas_em_pruned(Cov_x, B_inv, active_blocks)
        # update sigma
        Sigma0 = update_sigma_0(gammas, B)

        # convergence criterion
        mu_diff = jnp.abs(mu_x - state.mu_x)
        dmu = jnp.max(mu_diff)

        state = BSBL_State(
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



##################################################
# BSBL Bound Optimization
##################################################

def bsbl_bo(Phi, y, blk_len, 
    options: BSBL_Options = BSBL_Options()):
    """Reconstructs a block sparse signal using BSBL-BO algorithm

    Args:
        Phi (jax.numpy.ndarray): Sensing matrix
        y (jax.numpy.ndarray): Measurement vector
        blk_len (int): Length/size of each block
        options (BSBL_Options): Options for algorithm execution

    Returns:
        BSBL_State: Solution of the sparse recovery problem

    Note:
        * Phi must be a matrix. Linear operators are not supported
          since we need to break Phi down into submatrices for each block.
        * Use :py:func:`bsbl_bo_options` to initialize options for
          the algorithm.

    Examples:
        - :ref:`gallery:cs:bsbl:1`
    """
    
    # options
    learn_lambda = options.learn_lambda
    learn_block_corr = options.learn_block_corr
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
        state = BSBL_State(
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
        active_blocks = state.gammas > prune_gamma
        PhiBPhi = cum_phi_b_phi_pruned(Subdicts, state.Sigma0, active_blocks)
        H = compute_h(Phi, PhiBPhi, state.lambda_val)
        # posterior block means
        mu_x, Hy = compute_mu_x_pruned(state.Sigma0, H, y, active_blocks)
        # posterior block covariances
        Sigma_x, HPhi = compute_sigma_x_pruned(Phi, state.Sigma0, H, active_blocks)
        Cov_x = compute_cov_x(Sigma_x, mu_x)
        Bi_sum = compute_cov_x_sum_pruned(Cov_x, state.gammas, active_blocks)
        B, B_inv = compute_B_B_inv(Bi_sum)
        # flattened signal
        x_hat = mu_x.flatten()
        # residual
        res = y - Phi @ x_hat
        # residual norm squared
        r_norm_sqr = crn.sqr_norm_l2(res)
        # update lambda
        # lambda_val = update_lambda_rule_nojit(learn_lambda,
        #     state.lambda_val, Subdicts, state.gammas, Sigma_x,
        #     B_inv, r_norm_sqr, m)
        lambda_val = update_lambda_rule_jittable(learn_lambda,
            state.lambda_val, Subdicts, state.gammas, Sigma_x,
            B_inv, r_norm_sqr, m)
        # update gamma
        gammas = update_gammas_bo_pruned(state.gammas, B, Hy, HPhi, active_blocks)
        # update sigma
        Sigma0 = update_sigma_0(gammas, B)

        # convergence criterion
        mu_diff = jnp.abs(mu_x - state.mu_x)
        dmu = jnp.max(mu_diff)

        state = BSBL_State(
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

bsbl_bo_jit = jit(bsbl_bo, static_argnums=(2,))




def bsbl_bo_np(Phi, y, blk_len, 
    options: BSBL_Options = BSBL_Options()):
    """Reconstructs a block sparse signal using BSBL-BO algorithm

    Args:
        Phi (jax.numpy.ndarray): Sensing matrix
        y (jax.numpy.ndarray): Measurement vector
        blk_len (int): Length/size of each block
        options (BSBL_Options): Options for algorithm execution

    Returns:
        BSBL_State: Solution of the sparse recovery problem

    Note:
        * Phi must be a matrix. Linear operators are not supported
          since we need to break Phi down into submatrices for each block.
        * Use :py:func:`bsbl_bo_options` to initialize options for
          the algorithm.

    Examples:
        - :ref:`gallery:cs:bsbl:1`
    """
    
    # options
    learn_lambda = options.learn_lambda
    learn_block_corr = options.learn_block_corr
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
        state = BSBL_State(
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
        PhiBPhi = cum_phi_b_phi(Subdicts, state.Sigma0)
        H = compute_h(Phi, PhiBPhi, state.lambda_val)
        # posterior block means
        mu_x, Hy = compute_mu_x(state.Sigma0, H, y)
        # posterior block covariances
        Sigma_x, HPhi = compute_sigma_x(Phi, state.Sigma0, H)
        Cov_x = compute_cov_x(Sigma_x, mu_x)
        Bi_sum = compute_cov_x_sum(Cov_x, state.gammas)
        B, B_inv = compute_B_B_inv(Bi_sum)
        # flattened signal
        x_hat = mu_x.flatten()
        # residual
        res = y - Phi @ x_hat
        # residual norm squared
        r_norm_sqr = crn.sqr_norm_l2(res)
        # update lambda
        # lambda_val = update_lambda_rule_nojit(learn_lambda,
        #     state.lambda_val, Subdicts, state.gammas, Sigma_x,
        #     B_inv, r_norm_sqr, m)
        lambda_val = update_lambda_rule_jittable(learn_lambda,
            state.lambda_val, Subdicts, state.gammas, Sigma_x,
            B_inv, r_norm_sqr, m)
        # update gamma
        gammas = update_gammas_bo(state.gammas, B, Hy, HPhi)
        # update sigma
        Sigma0 = update_sigma_0(gammas, B)

        # convergence criterion
        mu_diff = jnp.abs(mu_x - state.mu_x)
        dmu = jnp.max(mu_diff)

        state = BSBL_State(
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

bsbl_bo_np_jit = jit(bsbl_bo_np, static_argnums=(2,))
