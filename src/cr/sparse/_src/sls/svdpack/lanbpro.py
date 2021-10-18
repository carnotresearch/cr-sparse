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

from .util import do_elr, do_elr_noop
from .reorth import reorth_mgs, reorth_noop

FUDGE = 1.01
M2 = 3/2
N2 = 3/2

KEY = random.PRNGKey(0)
KEYS = random.split(KEY, 20)

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


def lanbpro_init(A, k, p0, options: LanBDOptions):
    """Initialize the state with a starting vector 
    """
    # we follow steps from page 30 of Larsen paper
    m, n = A.shape
    U = jnp.zeros((m, k))
    V = jnp.zeros((n, k))
    alpha = jnp.zeros(k)
    beta = jnp.zeros(k+1)
    mu = jnp.zeros(k+1)
    nu = jnp.zeros(k+1)
    mumax = jnp.zeros(k)
    numax = jnp.zeros(k)
    indices = jnp.zeros(k, dtype=bool)
    b_fro = options.delta == 0
    # step 1
    p_norm = norm(p0)
    beta = beta.at[0].set(p_norm)
    u = crs.vec_safe_divide_by_scalar(p0, p_norm)
    U = U.at[:, 0].set(u)
    nu = nu.at[0].set(1)
    # step 2 r update
    r = A.trans(u)
    # step 2.1b alpha update
    r_norm = norm(r)
    alpha = alpha.at[0].set(r_norm)
    anorm = FUDGE * r_norm
    # step 2.1b v update
    v = crs.vec_safe_divide_by_scalar(r, r_norm)
    V = V.at[:, 0].set(v)
    # step 2.1b p update
    p = A.times(v) - alpha[0] * u
    # step 2.2b beta update
    p_norm = norm(p)
    beta = beta.at[1].set(p_norm)
    p, p_norm, p_proj = do_elr(u, p, p_norm, options.gamma)
    # update anorm estimate 
    anorm = jnp.maximum(anorm,FUDGE*jnp.hypot(alpha[0],beta[1]))
    # update mu for the first iteration before computation of U_1
    eps = jnp.finfo(float).eps
    eps1 = 100*eps/2
    T = eps1*(anorm + jnp.hypot(alpha[0],beta[1]) + jnp.hypot(alpha[0],beta[0]) )
    mu = mu.at[0].set(T / beta[1])
    mu = mu.at[1].set(1)
    # mumax update
    mumax = mumax.at[0].set(jnp.abs(mu[0]))
    # TODO add condition with elr > 0
    mu = mu.at[0].set(M2 * eps)
    # prepare state after completion of one iteration
    return  LanBProState(p=p, U=U, V=V, alpha=alpha, beta=beta,
        mu=mu, nu=nu, mumax=mumax, numax=numax,
        anorm=anorm,
        indices=indices, b_fro=b_fro, iterations=1,
        )

def lanbpro_random_start(key, A):
    """Creates a random starting vector for the algorithm"""
    m, n = A.shape
    r = random.uniform(key, (m,)) - 0.5
    return r


def lanbpro_iteration(A, state: LanBProState, options: LanBDOptions):
    """One single (j-th) iteration of Lanczos bidiagonalization with partial reorthogonalization algorithm
    """
    m, n = A.shape
    # copy variables from the state
    p = state.p
    U = state.U
    V = state.V
    alpha = state.alpha
    beta = state.beta
    mu = state.mu
    nu = state.nu
    mumax = state.mumax
    numax = state.numax
    saved_indices = state.indices
    anorm = state.anorm
    b_fro = state.b_fro
    b_force_reorth = state.b_force_reorth
    # the total number of iterations
    k = len(alpha)
    # index for k
    idx = jnp.arange(k)
    # iteration number 
    j = state.iterations
    # first j indices mask
    j_mask =  idx < j
    jp1_mask = idx <= j
    # options
    gamma = options.gamma
    elr = options.elr
    eps = options.eps
    delta = options.delta
    eta = options.eta
    # carry out the work for one iteration of lanbpro
    beta_j = beta[j]
    # compute next left singular vector
    u = crs.vec_safe_divide_by_scalar(p, beta_j)
    U = U.at[:, j].set(u)
    # step 2 r update
    v_jm1 = V[:, j-1]
    r = A.trans(u) - beta_j * v_jm1
    r_norm = norm(r)
    # update alpha_j
    alpha = alpha.at[j].set(r_norm)
    # elr condition
    b_no_fro = jnp.logical_not(b_fro)
    elr_cond = jnp.logical_and(jnp.logical_and(r_norm < gamma * beta_j, elr), b_no_fro)
    # extended local reorthogonalization of r w.r.t. previous v_j
    r, r_norm, proj = lax.cond(elr_cond, 
        lambda _ : do_elr(v_jm1, r, r_norm, gamma),
        lambda _ : do_elr_noop(r, r_norm),
        None
    )
    # save updated r_norm in alpha_j (if there are any changes)
    alpha = alpha.at[j].set(r_norm)
    # make changes to beta_j if required.
    beta = beta.at[j].add(proj)
    # norm estimate
    anorm_up_1 = lambda anorm: jnp.maximum(anorm,FUDGE*jnp.sqrt(alpha[0]**2+beta[1]**2+alpha[1]*beta[1]))
    anorm_up_j = lambda anorm: jnp.maximum(anorm,FUDGE*jnp.sqrt(alpha[j-1]**2+beta[j]**2+alpha[j-1]*
	    beta[j-1] + alpha[j]*beta[j]))
    anorm = lax.cond(j == 1, anorm_up_1, anorm_up_j, anorm)
    # nu update condition
    nu_update_cond = jnp.logical_and(b_no_fro, r_norm != 0)
    nu, numax = lax.cond(nu_update_cond, 
        lambda nu: update_nu(nu, numax, mu, j, alpha, beta, anorm),
        lambda nu : (nu, numax),
        nu
    )
    # TODO add condition with elr > 0
    nu = nu.at[j-1].set(N2 * eps)
    # condition for partial or full reorthogonalization
    reorth_cond = jnp.logical_or(b_fro, numax[j] > delta)
    reorth_cond = jnp.logical_or(reorth_cond, b_force_reorth)
    reorth_cond = jnp.logical_and(reorth_cond, alpha[j] != 0)
    # identify the indices at which partial or full reorthogonalization will be done
    indices = lax.cond(jnp.logical_or(b_fro, eta == 0),
        # full reorthogonalization case
        lambda _ : j_mask,
        # partial reorthogonalization case
        lambda _ : lax.cond(b_force_reorth,
            lambda _ : saved_indices,
            lambda _ : compute_ind(nu[:k].at[j].set(0), delta, eta),
            None
        ),
        None
    )
    # reorthogonalize r if required
    r, r_norm, iters = lax.cond(reorth_cond,
        lambda _ : reorth_mgs(V, r, r_norm, indices, gamma),
        lambda _ : reorth_noop(r, r_norm),
        None
    )
    # update alpha_j again if required
    alpha = alpha.at[j].set(r_norm)
    # reset nu in the entries which have been reorthogonalized
    nu_indices = jnp.append(indices, jnp.zeros(len(nu) - len(indices), dtype=bool))
    nu = jnp.where(nu_indices, N2*eps, nu)
    # change the b_force_reorth flag if needed
    b_force_reorth = jnp.logical_xor(reorth_cond, b_force_reorth)
    # TODO we need to replace r with a new unit vector if r is close to 0.
    # this is the case where r is in the column space of previous V vectors
    # step 2.1b v update
    v = crs.vec_safe_divide_by_scalar(r, r_norm)
    V = V.at[:, j].set(v)
    # Lanczos step to generate u_{j+1}
    # step 2.1b p update
    p = A.times(v) - alpha[j] * u
    # step 2.2b beta update
    p_norm = norm(p)
    beta = beta.at[j+1].set(p_norm)
    # elr condition for p
    elr_cond = jnp.logical_and(jnp.logical_and(p_norm < gamma * r_norm, elr), b_no_fro)
    # extended local reorthogonalization of p w.r.t. previous u_j
    p, p_norm, proj = lax.cond(elr_cond, 
        lambda _ : do_elr(u, p, p_norm, gamma),
        lambda _ : do_elr_noop(p, p_norm),
        None
    )
    # save updated p_norm in beta_{j+1} (if there are any changes)
    beta = beta.at[j+1].set(p_norm)
    # make changes to alpha_j if required.
    alpha = alpha.at[j].add(proj)
    anorm = jnp.maximum(anorm,FUDGE*jnp.sqrt(alpha[j]**2+beta[j+1]**2+alpha[j]*beta[j]))

    # mu update condition
    mu_update_cond = jnp.logical_and(b_no_fro, p_norm != 0)
    mu, mumax = lax.cond(mu_update_cond, 
        lambda mu: update_mu(mu, mumax, nu, j, alpha, beta, anorm),
        lambda mu : (mu, mumax),
        mu
    )
    # TODO add condition with elr > 0
    mu = mu.at[j].set(M2 * eps)
    # condition for partial or full reorthogonalization
    reorth_cond = jnp.logical_or(b_fro, mumax[j] > delta)
    reorth_cond = jnp.logical_or(reorth_cond, b_force_reorth)
    reorth_cond = jnp.logical_and(reorth_cond, p_norm != 0)
    # identify the indices at which partial or full reorthogonalization will be done
    # from U_0 to U_j [j+1 vectors] have already been computed 
    indices = lax.cond(jnp.logical_or(b_fro, eta == 0),
        # full reorthogonalization case
        lambda _ : jp1_mask,
        # partial reorthogonalization case
        lambda _ : lax.cond(b_force_reorth,
            lambda _ : saved_indices,
            lambda _ : compute_ind(mu[:k].at[j+1].set(0), delta, eta),
            None
        ),
        None
    )
    # reorthogonalize p if required
    p, p_norm, iters = lax.cond(reorth_cond,
        lambda _ : reorth_mgs(U, p, p_norm, indices, gamma),
        lambda _ : reorth_noop(p, p_norm),
        None
    )
    # save updated p_norm in beta_{j+1} (if there are any changes)
    beta = beta.at[j+1].set(p_norm)
    # reset nu in the entries which have been reorthogonalized
    mu_indices = jnp.append(indices, jnp.zeros(len(mu) - len(indices), dtype=bool))
    mu = jnp.where(mu_indices, M2*eps, mu)
    # change the b_force_reorth flag if needed
    b_force_reorth = jnp.logical_xor(reorth_cond, b_force_reorth)
    # TODO we need to replace p with a new unit vector if p is close to 0.


    # save the indices back for next iteration if b_force_reorth
    saved_indices = saved_indices.at[:len(indices)].set(indices)

    # prepare the state for next iteration
    return  LanBProState(p=p, U=U, V=V, alpha=alpha, beta=beta,
        mu=mu, nu=nu, mumax=mumax, numax=numax,
        anorm=anorm,
        indices=saved_indices, b_fro=b_fro, b_force_reorth=b_force_reorth,
        iterations=j+1
        )


lanbpro_iteration_jit = jit(lanbpro_iteration, static_argnums=(0,))


def lanbpro(A, k, p0):
    """K steps of the  Lanczos bidiagonalization with partial reorthogonalization
    """
    options = lanbpro_options_init(k)
    state  = lanbpro_init(A, k, p0, options)

    def cond(state):
        return state.iterations < k

    def body(state):
        state =  lanbpro_iteration(A, state, options, state.iterations)
        return state

    # state = lax.while_loop(cond, body, state)
    # while cond(state):
    #     state = body(state)
    state = lax.fori_loop(1, k, 
        lambda i, state: lanbpro_iteration(A, state, options),
        state)
    return state

lanbpro_jit = jit(lanbpro, static_argnums=(0, 1))

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
