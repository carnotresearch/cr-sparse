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
"""

from jax import vmap, lax
from jax.lax import fori_loop
import jax.numpy as jnp

import cr.nimble as crn
import cr.sparse.block.block as crblock


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

def cum_phi_b_phi_pruned(Phi, Sigma0, active_blocks):
    n_blocks = len(Sigma0)
    m, n = Phi.shape
    # block length
    b = n // n_blocks
    starts = [i*b for i in range(n_blocks)]

    result = jnp.zeros((m, m))
    # zero value
    z = result
    for i in range(n_blocks):
        result += lax.cond(active_blocks[i],
            lambda _: phi_b_phi(Phi, starts[i], b, Sigma0[i]),
            lambda _: z,
            None)
    return result


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
    return mu_x

def compute_sigma_x(Phi, Sigma0, H):
    # block length
    b = Sigma0.shape[1]
    HPhi = H @ Phi
    # Extract the block diagonals
    blocks = crn.block_diag(HPhi, b)
    Sigma_x = vmap(
        lambda A, B: A - A  @ B @ A,
        in_axes=(0, 0))(Sigma0, blocks)
    return Sigma_x

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

def compute_B_B_inv(B0):
    m0 = jnp.mean(jnp.diag(B0))
    m1 = jnp.mean(jnp.diag(B0, 1))
    # AR-1 coefficient
    r = m1 / m0
    # print(f'r: {r}')
    # block size
    b = B0.shape[0]
    c = r ** jnp.arange(b)
    B = crn.toeplitz_mat(c, c)
    B_inv = jnp.linalg.inv(B)
    return B, B_inv


def update_lambda_high_snr(gammas, Sigma_x, B_inv, res):
    ll = vmap(
        lambda g, sx: jnp.trace(sx @ B_inv) / g,
        in_axes=(0, 0))(gammas, Sigma_x)
    lambda_comp = jnp.sum(ll)


def update_gammas(Cov_x, B_inv):
    n_blocks, blk_size, _ = Cov_x.shape
    gammas = vmap(
        lambda cx: jnp.trace(B_inv @ cx))(Cov_x)
    gammas = gammas / blk_size
    return gammas

def update_sigma_0(gammas, B):
    Sigma0 = vmap(
        lambda gamma: gamma * B,
        in_axes=(0,))(gammas)
    return Sigma0