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
    eta = (eps ** 0.75) / jnp.sqrt(k)
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
    anorms : jnp.ndarray
    "Tracking the a norms over multiple iterations" 
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
        j = self.iterations
        # s.append(f'p= {self.p}')
        # s.append(f'U= {self.U}')
        # s.append(f'V= {self.V}')
        s.append(f'alpha= {self.alpha[:j]}')
        s.append(f'beta= {self.beta[:j+1]}')
        s.append(f'mu= {self.mu[:j]}')
        s.append(f'nu= {self.nu[:j]}')
        s.append(f'mumax= {self.mumax[:j]}')
        s.append(f'numax= {self.numax[:j]}')
        # s.append(f'anorm= {self.anorm}')
        s.append(f'b_fro= {self.b_fro}, b_force_reorth={self.b_force_reorth}')
        ind = jnp.where(self.indices)[0]
        s.append(f'indices= {ind}')
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
        mask = jnp.logical_and(idx > 0, idx < j)
        # alpha between 1 to j-1
        alpha0 = jnp.where(mask, alpha, 0)
        # beta between 1 to j-1
        beta0 = beta[:k]
        beta0 = jnp.where(mask, beta0, 0)
        # nu between 1 to j-1
        nu0 = jnp.where(mask, nu, 0)
        # nu between 0 to j
        nu1 = jnp.roll(nu, 1)
        nu1 = jnp.where(mask, nu1, 0)
        # mu between 1 to j-1
        mu0 = jnp.where(mask, mu, 0)
        tmp = alpha0*nu0 + beta0*nu1 - aj*mu0
        y = jnp.sqrt(alpha0**2+beta0**2)
        T = eps1*(x + y + anorm)
        tmp = binv*(tmp + jnp.sign(tmp)*T)
        # update mu on the range 1 to j-1
        mu = jnp.where(mask, tmp, mu)
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
    ainv = 1/alpha[j]
    eps1 = 100*eps/2
    aj = alpha[j]
    bj = beta[j]
    x = jnp.sqrt(aj**2 + bj**2)
    k = len(alpha)
    idx = jnp.arange(k)
    j_mask = idx < j
    # alpha from 0 to j-1
    alpha =  jnp.where(j_mask, alpha, 0)
    # beta from 1 to j
    beta = beta[1:]
    beta = jnp.where(j_mask, beta, 0)
    # mu from 0 to j-1
    mu0 = jnp.where(j_mask, mu, 0)
    # nu from 0 to j-1
    nu0 = jnp.where(j_mask, nu, 0)
    # mu from 1 to j
    mu1 = jnp.roll(mu, -1)
    mu1 = jnp.where(j_mask, mu1, 0)
    # compute T
    y = jnp.sqrt(alpha**2 + beta**2)
    T = eps1*(x + y + anorm)
    # update nu
    nu0 = beta*mu1 + alpha*mu0 - bj*nu0
    nu0 =  ainv*(nu0 + jnp.sign(nu0)*T)
    numax = numax.at[j].set(jnp.max(jnp.abs(nu0) ))
    return nu0, numax


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
    mu = jnp.abs(mu)
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


def bpro_norm_estimate(alpha, beta):
    """Estimates the norm based on a 6x5 matrix
    """
    k = 5
    if len(alpha) < k:
        alpha = jnp.pad(alpha, (0, k - len(alpha)))
    if len(beta) <= k:
        beta = jnp.pad(beta, (0, k + 1 - len(beta)))
    # prepare the k+1 x k bidiagonal matrix
    B = jnp.zeros((k+1, k))
    # diagonal indices for k alpha entries
    indices = jnp.diag_indices(k)
    B = B.at[indices].set(alpha[:k])
    # subdiagonal indices for k beta entries (from second row)
    rows, cols = indices
    rows = rows + 1
    B = B.at[(rows, cols)].set(beta[1:k+1])
    result = norm(B, 2)
    return result

def do_elr(v_prev, v, v_norm, gamma):
    """
    Extended local reorthogonalization
    """
    def init():
        t = jnp.vdot(v_prev, v)
        v2 = v - t * v_prev
        v2_norm = norm(v2)
        return v2, v2_norm, v_norm, t

    def body(state):
        v, old_norm, older_norm, proj = state
        t = jnp.vdot(v_prev, v)
        v = v - t * v_prev
        v_norm = norm(v)
        return v, v_norm, old_norm, proj + t

    def cond(state):
        v, v_norm, old_norm, proj = state
        return v_norm < gamma * old_norm

    # state = init()
    # while cond(state):
    #     state = body(state)
    state = lax.while_loop(cond, body, init())
    v, v_norm, old_norm, proj = state
    return v, v_norm, proj
