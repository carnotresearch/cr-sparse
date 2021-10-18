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

FUDGE = 1.01
M2 = 3/2
N2 = 3/2

class LanBDOptions(NamedTuple):
    """Options for Lanczos bidiagonalization with partial reorthogonalization algorithm"""
    delta : float = 0.
    "desired level of orthogonality"
    eta : float = 0.
    "desired level of orthogonality after reorthogonalization"
    gamma : float = 0.
    "Tolerance for iterate Gram-Schmidt"
    cgs : bool = False
    "Flag for Classic / Modified Gram Schmidt"
    elr : int = 2
    "Flag for extended local reorthogonalization"
    eps : float = jnp.finfo(float).eps
    "eps value"

def lanbpro_options_init(k : int, eps: float =jnp.finfo(float).eps):
    """Initializes options"""
    delta = jnp.sqrt(eps/k)
    eta = eps ** 0.75 / jnp.sqrt(k)
    gamma = 1 / math.sqrt(2)
    return LanBDOptions(delta=delta, eta=eta, gamma=gamma, eps=eps)

class LanBProState(NamedTuple):
    """State of Lanczos bidiagonalization with partial reorthogonalization algorithm

    At the end of j-iterations
    - alpha has j entries
    - U has j columns
    - V has j columns
    - beta has j+1 entries
    """
    p : jnp.ndarray
    "Starting vector for next left singular vector"
    U : jnp.ndarray
    "Left singular vectors"
    V : jnp.ndarray
    "Right singular vectors"
    alpha: jnp.ndarray
    "Diagoanal elements of B"
    beta: jnp.ndarray
    "Subdiagonal elements of B"
    mu: jnp.ndarray
    "Tracker for U inner products"
    nu: jnp.ndarray
    "Tracker for V inner products"
    mumax: jnp.ndarray
    "Maximum values of mu for each iteration"
    numax: jnp.ndarray
    "Maximum values of nu for each iteration"
    indices: jnp.ndarray
    "Set of indices to orthogonalize against"
    anorm : float = -1.
    "Running estimate of the norm of A"    
    nreorthu: int = 0
    "Number of reorthogonalizations on U"
    nreorthv: int = 0
    "Number of reorthogonalizations on V"
    npu: int = 0
    "Number of U inner products"
    npv: int = 0
    "Number of V inner products"
    nrenewu: int = 0
    "Number of times u vectors were renewed"
    nrenewv: int = 0
    "Number of times v vectors were renewed"
    b_force_reorth: bool = False
    "Indicates if reorthogonalization is forced in next iter"
    b_fro: bool = False
    "flag to indicate if full reorthogonalization is to be done"
    b_est_anorm: bool = True
    "Flag which indicates whether norm of A is to be estimated"
    iterations: int = 0
    "iteration number for j-th Lanczos vector"

    def __str__(self):
        s = []
        s.append(f'p= {self.p}')
        s.append(f'U= {self.U}')
        s.append(f'V= {self.V}')
        s.append(f'alpha= {self.alpha}')
        s.append(f'beta= {self.beta}')
        s.append(f'mu= {self.mu}')
        s.append(f'nu= {self.nu}')
        s.append(f'mumax= {self.mumax}')
        s.append(f'numax= {self.numax}')
        s.append(f'indices= {self.indices}')
        s.append(f'anorm= {self.anorm}')
        s.append(f'iterations= {self.iterations}')
        return '\n'.join(s)

def lanbpro_random_start(key, A):
    """Creates a random starting vector for the algorithm"""
    m, n = A.shape
    r = random.uniform(key, (m,)) - 0.5
    return r

def update_mu(mu, mumax, nu, j, alpha, beta, anorm):
    """Updates mu

    At the beginning of this call U[j+1] is being computed.

    * p has been computed
    * alpha[j] has been computed
    * j+1 U columns are already computed.
    * j+1 V columns are already computed.
    * beta[j+1] has been computed.
    * Estimates of inner product of U[j+1] with U_0 till U_j is being updated.s

    For U_1, 1 entry will be updated.
    For U_2, 2 entries will be updated.

    Essentially mu[0 till j] will be updated.
    mu[j+1] will always be set to 1.

    The mu update is broken down into four parts:

    * update mu[0]
    * update mu[0:j]
    * update mu[j]
    * update mu[j+1]
    """
    eps = jnp.finfo(float).eps
    bjp1 = beta[j+1]
    binv = 1/bjp1
    eps1 = 100*eps/2
    a0 = alpha[0]
    b0 = beta[0]
    aj = alpha[j]
    bj = beta[j]
    k = len(alpha)
    # mu[0] calculation
    tmp = a0*nu[0] - aj*mu[0]
    x = jnp.sqrt(aj**2+bjp1**2)
    y = jnp.sqrt(a0**2+b0**2)
    T = eps1*(x + y + anorm)
    tmp = binv * (tmp + jnp.sign(tmp)*T)
    mu = mu.at[0].set(tmp)
    # sub function to update the middle part of mu
    def update_middle_part(mu):
        idx = jnp.arange(k)
        mask_1 = jnp.logical_and(idx > 0, idx < j)
        mask_0 = idx < j-1
        alpha_sub = jnp.where(mask_1, alpha, 0)
        beta_base = beta[:k]
        beta_sub = jnp.where(mask_1, beta_base, 0)
        nu_base = nu[:k]
        nu_sub = jnp.where(mask_1, nu_base, 0)
        nu_sub_m1 = jnp.where(mask_0, nu_base, 0)
        mu_base = mu[:k]
        mu_sub = jnp.where(mask_1, mu_base, 0)
        # relevant part of alpha
        # k = slice(1, j)
        # relevant part of beta
        # km1 = slice(0, j-1)
        tmp = alpha_sub*nu_sub + beta_sub*nu_sub_m1 - aj*mu_sub
        y = jnp.sqrt(alpha_sub**2+beta_sub**2)
        T = eps1*(x + y + anorm)
        tmp = binv*(tmp + jnp.sign(tmp)*T)
        mu = mu.at[:k].set(tmp)
        return mu
    # update the middle part only if j > 1
    mu = lax.cond(j > 1, update_middle_part, lambda mu: mu, mu)
    # update mu[j]
    y = jnp.sqrt(aj**2+bj**2)
    T = eps1*(x + y + anorm)
    tmp = bj*nu[j-1]
    mu = mu.at[j].set(binv * (tmp + jnp.sign(tmp)*T))
    # mumax update
    mumax = mumax.at[j].set(jnp.max(jnp.abs(mu) ))
    # update mu[j+1] to 1
    mu = mu.at[j+1].set(1)
    return mu, mumax

def update_nu(nu, numax, mu, j, alpha, beta, anorm):
    """Updates nu

    At the beginning of this call V_j is being computed.
 
    * r has been computed
    * alpha[j] has been computed
    * j+1 U columns are already computed.
    * j V columns are already computed.
    * beta[j] has been computed.
    * Estimates of inner product of V[j] with V_0 till V_{j-1} is being updated.
    * Thus, total j entries are under consideration for computing the max value.
    * nu[j] = 1 always.
    """
    eps = jnp.finfo(float).eps
    #print(eps)
    ainv = 1/alpha[j]
    #print(ainv)
    eps1 = 100*eps/2
    last = jnp.sqrt(alpha[j]**2 + beta[j]**2)
    #print(last)
    k = len(alpha)
    idx = jnp.arange(k)
    j_mask = idx < j
    alpha_sub =  jnp.where(j_mask, alpha, 0)
    beta_rest = beta[1:]
    beta_sub = jnp.where(j_mask, beta_rest, 0)
    mu_base = mu[:k]
    mu_sub = jnp.where(j_mask, mu_base, 0)
    nu_base = nu[:k]
    nu_sub = jnp.where(j_mask, nu_base, 0)
    mu_rest = mu[1:]
    mu_sub1 = jnp.where(j_mask, mu_rest, 0)

    y = jnp.sqrt(alpha_sub**2 + beta_sub**2)
    T = eps1*(y + last + anorm)
    # print(T)
    nu_sub = beta_sub*mu_sub1 + alpha_sub*mu_sub - beta[j]*nu_sub
    nu_sub =  ainv*(nu_sub + jnp.sign(nu_sub)*T)
    # numax update
    numax = numax.at[j].set(jnp.max(jnp.abs(nu_sub) ))

    nu = nu.at[:k].set(nu_sub)
    nu = nu.at[j].set(1.)
    return nu, numax


def abs_max_boolean_index(mu):
    """Returns a boolean index array which is True only at the place of 
    absolute maximum
    """
    return jnp.zeros(len(mu), dtype=bool).at[jnp.argmax(jnp.abs(mu))].set(True)


def update_r(mu, indices, i, eta):
    """Start from i, go back and set all consecutive entries to True which are above eta.
    """
    state = indices, i-1
    def cond(state):
        indices, j = state
        c = jnp.logical_and(j>=0, mu[j] >= eta)
        c = jnp.logical_and(c, indices[j] == False)
        return c

    def body(state):
        indices, j = state
        indices = indices.at[j].set(True)
        return indices, j-1

    return lax.while_loop(cond, body, state)[0]
    # while cond(state):
    #     state = body(state)
    # return state[0]

def update_s(mu, indices, i, eta):
    """Start from i, go forward and set all consecutive entries to True which are above eta.
    """
    n = len(mu)
    state = indices, i+1
    def cond(state):
        indices, j = state
        c = jnp.logical_and(j<n, mu[j] >= eta)
        c = jnp.logical_and(c, indices[j] == False)
        return c

    def body(state):
        indices, j = state
        indices = indices.at[j].set(True)
        return indices, j+1

    return lax.while_loop(cond, body, state)[0]
    # while cond(state):
    #     state = body(state)
    # return state[0]

def extend_around(mu, indices, i, eta):
    indices = update_r(mu, indices, i, eta)
    indices = update_s(mu, indices, i, eta)
    return indices

def compute_ind(mu, delta, eta):
    """Identifies indices for reorthogonalization
    """
    k = len(mu)
    indices = jnp.zeros(k, dtype=bool)
    indices = mu >= delta
    # number of indices to orthogonalize against
    n = jnp.sum(indices)
    indices = lax.cond(n > 0, lambda _ : indices, lambda _ : abs_max_boolean_index(mu), None)
    # make a copy
    indices2 = indices
    # loop for extending reorthogonalization around already identified indices
    def extend_rs_iter(i, indices):
        return lax.cond(indices2[i], lambda indices :  extend_around(mu, indices, i, eta), 
            lambda indices: indices, indices)
    indices = lax.fori_loop(0, k, extend_rs_iter, indices)
    return indices
