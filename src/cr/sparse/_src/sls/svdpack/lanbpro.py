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

from jax import lax, jit, vmap, random
import jax.numpy as jnp
from jax.numpy.linalg import norm


import cr.sparse as crs

from .util import do_elr, do_elr_noop
from cr.sparse.la.svd import reorth_mgs, reorth_noop
from cr.sparse.la.svd import (
    LanBDOptions,
    LanBProState,
    lanbpro_options_init,
    update_mu,
    update_nu,
    compute_ind
)

FUDGE = 1.01
M2 = 3/2
N2 = 3/2

KEY = random.PRNGKey(0)
KEYS = random.split(KEY, 20)


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


