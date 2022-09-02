# Copyright 2021 CR-Suite Development Team
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


import cr.nimble as cnb

from cr.nimble.svd import reorth_mgs, reorth_noop
from cr.nimble.svd import (
    LanBDOptions,
    LanBProState,
    lanbpro_options_init,
    do_elr,
    update_mu,
    update_nu,
    compute_ind,
    bpro_norm_estimate
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
    mu = jnp.zeros(k)
    nu = jnp.zeros(k)
    mumax = jnp.zeros(k)
    numax = jnp.zeros(k)
    indices = jnp.zeros(k, dtype=bool)
    anorms = jnp.zeros(k)
    # options 
    delta = options.delta
    eps = options.eps
    gamma = options.gamma
    # whether full reorthogonalization will be done
    b_fro = delta == 0
    # initial value of force reorthogonalization
    b_force_reorth = False
    # step 1
    p_norm = norm(p0)
    # beta_0
    beta = beta.at[0].set(p_norm)
    # U_0
    u = cnb.vec_safe_divide_by_scalar(p0, p_norm)
    U = U.at[:, 0].set(u)
    # step 2 r update
    r = A.trans(u)
    # step 2.1b alpha update
    r_norm = norm(r)
    alpha = alpha.at[0].set(r_norm)
    anorm = FUDGE * r_norm
    # step 2.1b v update
    v = cnb.vec_safe_divide_by_scalar(r, r_norm)
    V = V.at[:, 0].set(v)
    # step 2.1b p update
    p = A.times(v) - alpha[0] * u
    # step 2.2b beta update
    p_norm = norm(p)
    p, p_norm, p_proj = do_elr(u, p, p_norm, options.gamma)
    # Check for convergence or failure to maintain semiorthogonality
    semiorth_cond = p_norm < max(m, n) * anorm * eps
    p, p_norm, b_force_reorth, indices = lax.cond(semiorth_cond, 
        # compute a new random p vector orthogonal to previous U
        lambda _ : (new_p_vec(A, U, 1, gamma)[0], 0., True, indices.at[0].set(True)),
        lambda _ : (p, p_norm, b_force_reorth, indices), 
        None) 
    beta = beta.at[1].set(p_norm)
    # update anorm estimate 
    anorm = jnp.maximum(anorm,FUDGE*jnp.hypot(alpha[0],beta[1]))
    # update mu for the first iteration before computation of U_1
    eps = jnp.finfo(float).eps
    eps1 = 100*eps/2
    T = eps1*(anorm + jnp.hypot(alpha[0],beta[1]) + jnp.hypot(alpha[0],beta[0]) )
    # TODO this is problematic if p_norm is 0
    mu0 = lax.cond(p_norm, lambda _ :  T / p_norm, lambda _ : 0., None)
    mu = mu.at[0].set(mu0)
    # mumax update
    mumax = mumax.at[0].set(jnp.abs(mu[0]))
    # TODO add condition with elr > 0
    mu = mu.at[0].set(M2 * eps)
    # prepare state after completion of one iteration
    anorms = anorms.at[0].set(anorm)
    return  LanBProState(p=p, U=U, V=V, alpha=alpha, beta=beta,
        mu=mu, nu=nu, mumax=mumax, numax=numax,
        anorm=anorm, anorms=anorms,
        indices=indices, 
        b_force_reorth=b_force_reorth, b_fro=b_fro, iterations=1,
        )


def lanbpro_iteration(A, state: LanBProState, options: LanBDOptions):
    """One single (j-th) iteration of Lanczos bidiagonalization with partial reorthogonalization algorithm
    """
    m, n = A.shape
    max_m_n = max(m, n)
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
    indices = state.indices
    anorm = state.anorm
    b_fro = state.b_fro
    b_force_reorth = state.b_force_reorth
    b_est_anorm = state.b_est_anorm
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
    u = cnb.vec_safe_divide_by_scalar(p, beta_j)
    U = U.at[:, j].set(u)
    # once sufficient iterations are completed, we can
    # compute a better estimate of a norm
    anorm, b_est_anorm = lax.cond(j == 5, 
        # Replace norm estimate with largest Ritz value.
        lambda _ : (FUDGE*bpro_norm_estimate(alpha, beta), False),
        # continue with current value
        lambda _ : (anorm, b_est_anorm),
        None
    )
    # step 2 r update
    v_jm1 = V[:, j-1]
    r = A.trans(u) - beta_j * v_jm1
    r_norm = norm(r)
    # elr condition
    b_no_fro = jnp.logical_not(b_fro)
    elr_cond = jnp.logical_and(jnp.logical_and(r_norm < gamma * beta_j, elr), b_no_fro)
    # extended local reorthogonalization of r w.r.t. previous v_j
    r, r_norm, proj = lax.cond(elr_cond, 
        lambda _ : do_elr(v_jm1, r, r_norm, gamma),
        lambda _ : (r, r_norm, 0.),
        None
    )
    # save updated r_norm in alpha_j
    alpha = alpha.at[j].set(r_norm)
    # make changes to beta_j if required.
    beta = beta.at[j].add(proj)
    # norm estimate
    anorm_up_1 = lambda anorm: jnp.maximum(anorm,
        FUDGE*jnp.sqrt(alpha[0]**2+beta[1]**2+alpha[1]*beta[1]))
    anorm_up_j = lambda anorm: jnp.maximum(anorm,
        FUDGE*jnp.sqrt(alpha[j-1]**2+beta[j]**2
            + alpha[j-1]*beta[j-1] + alpha[j]*beta[j]))
    anorm = lax.cond(b_est_anorm,
        # We need to update norm estimate
        lambda anorm : lax.cond(j == 1, anorm_up_1, anorm_up_j, anorm),
        # no more norm estimation needed
        lambda anorm : anorm,
        anorm)
    # nu update condition
    nu_update_cond = jnp.logical_and(b_no_fro, r_norm != 0)
    nu, numax = lax.cond(nu_update_cond, 
        lambda nu: update_nu(nu, numax, mu, j, alpha, beta, anorm),
        lambda nu : (nu, numax),
        nu
    )
    # if elr is on, then current vector is orthogonalized against previous one
    nu = lax.cond(elr > 0,
        lambda nu: nu.at[j-1].set(N2 * eps),
        lambda nu: nu,
        nu)
    # condition for partial or full reorthogonalization
    reorth_cond = jnp.logical_or(b_fro, numax[j] > delta)
    reorth_cond = jnp.logical_or(reorth_cond, b_force_reorth)
    reorth_cond = jnp.logical_and(reorth_cond, alpha[j] != 0)
    # function to reorth r w.r.t. previous V
    def reorth_r(_):
        # identify the indices at which partial or full reorthogonalization will be done
        reorth_indices = lax.cond(jnp.logical_or(b_fro, eta == 0),
            # full reorthogonalization case
            lambda _ : j_mask,
            # partial reorthogonalization case
            lambda _ : lax.cond(b_force_reorth,
                lambda _ : indices,
                lambda _ : compute_ind(nu, delta, eta),
                None
            ),
            None
        )
        # carry out reorthogonalization
        r2, r2_norm, iters = reorth_mgs(V, r, r_norm, reorth_indices, gamma)
        # reset nu in the entries which have been reorthogonalized
        nu2 = jnp.where(reorth_indices, N2*eps, nu)
        # if a reorthogonalization was forced. it won't be for next iteration
        b_force_reorth2 = jnp.logical_not(b_force_reorth)
        return r2, r2_norm, reorth_indices, nu2, b_force_reorth2
    # reorthogonalize r if required
    r, r_norm, indices, nu, b_force_reorth = lax.cond(reorth_cond,
        reorth_r,
        lambda _ : (r, r_norm, indices, nu, b_force_reorth),
        None
    )
    # Check for convergence or failure to maintain semiorthogonality
    # this is the case where r is in the column space of previous V vectors
    semiorth_cond = r_norm < max(m, n) * anorm * eps
    r, r_norm, b_force_reorth, indices = lax.cond(semiorth_cond, 
        # compute a new random r vector orthogonal to previous V
        lambda _ : (new_r_vec(A, V, j, gamma)[0], 0., True, j_mask),
        lambda _ : (r, r_norm, b_force_reorth, indices), 
        None) 
    # update alpha_j again if required
    alpha = alpha.at[j].set(r_norm)
    # step 2.1b v update
    v = cnb.vec_safe_divide_by_scalar(r, r_norm)
    V = V.at[:, j].set(v)
    # Lanczos step to generate u_{j+1}
    # step 2.1b p update
    p = A.times(v) - alpha[j] * u
    # step 2.2b beta update
    p_norm = norm(p)
    # elr condition for p
    elr_cond = jnp.logical_and(jnp.logical_and(p_norm < gamma * r_norm, elr), b_no_fro)
    # extended local reorthogonalization of p w.r.t. previous u_j
    p, p_norm, proj = lax.cond(elr_cond, 
        lambda _ : do_elr(u, p, p_norm, gamma),
        lambda _ : (p, p_norm, 0.),
        None
    )
    # save updated p_norm in beta_{j+1} (if there are any changes)
    beta = beta.at[j+1].set(p_norm)
    # make changes to alpha_j if required.
    alpha = alpha.at[j].add(proj)
    anorm = lax.cond(b_est_anorm,
        # we need to update anorm estimate
        lambda anorm: jnp.maximum(anorm,FUDGE*jnp.sqrt(alpha[j]**2+beta[j+1]**2+alpha[j]*beta[j])),
        # no further need to update norm estimate
        lambda anorm: anorm,
        anorm)

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
    # function to reorth p w.r.t. previous U
    def reorth_p(_):
        # identify the indices at which partial or full reorthogonalization will be done
        # from U_0 to U_j [j+1 vectors] have already been computed 
        reorth_indices = lax.cond(jnp.logical_or(b_fro, eta == 0),
            # full reorthogonalization case
            lambda _ : jp1_mask,
            # partial reorthogonalization case
            lambda _ : lax.cond(b_force_reorth,
                # for forced reorth, we need to add one more vec
                lambda _ : indices.at[k - jnp.argmax(indices[::-1])].set(True),
                lambda _ : compute_ind(mu, delta, eta),
                None
            ),
            None
        )
        # carry out reorthogonalization
        p2, p2_norm, iters = reorth_mgs(U, p, p_norm, indices, gamma)
        # reset mu in the entries which have been reorthogonalized
        mu2 = jnp.where(reorth_indices, M2*eps, mu)
        # if a reorthogonalization was forced. it won't be for next iteration
        b_force_reorth2 = jnp.logical_not(b_force_reorth)
        return p2, p2_norm, reorth_indices, mu2, b_force_reorth2
    # reorthogonalize p if required
    p, p_norm, indices, nu, b_force_reorth = lax.cond(reorth_cond,
        reorth_p,
        lambda _ : (p, p_norm, indices, nu, b_force_reorth),
        None
    )
    # Check for convergence or failure to maintain semiorthogonality
    semiorth_cond = p_norm < max(m, n) * anorm * eps
    p, p_norm, b_force_reorth, indices = lax.cond(semiorth_cond, 
        # compute a new random p vector orthogonal to previous U
        lambda _ : (new_p_vec(A, U, j+1, gamma)[0], 0., True, jp1_mask),
        lambda _ : (p, p_norm, b_force_reorth, indices), 
        None) 
    # save updated p_norm in beta_{j+1} (if there are any changes)
    beta = beta.at[j+1].set(p_norm)

    # track anorm for the current iteration
    anorms = state.anorms.at[j].set(anorm)
    # prepare the state for next iteration
    return  LanBProState(p=p, U=U, V=V, alpha=alpha, beta=beta,
        mu=mu, nu=nu, mumax=mumax, numax=numax,
        anorm=anorm, anorms=anorms,
        indices=indices, b_fro=b_fro, b_force_reorth=b_force_reorth,
        b_est_anorm=b_est_anorm, iterations=j+1
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


def new_p_vec(A, U, j,  gamma):
    """Generates a new p vector which is orthogonal to all previous U vectors
    """
    m, n = A.shape
    idx = jnp.arange(U.shape[1])
    j_mask =  idx < j

    def p_vec(i):
        p = random.uniform(KEYS[i], (n,))
        p = A.times(p)
        p_norm = norm(p)
        p, p_norm, iters = reorth_mgs(U, p, p_norm, j_mask, gamma)
        return p, p_norm

    def init():
        p, p_norm = p_vec(0)
        return p, p_norm, 1

    def cond(state):
        p, p_norm, iterations = state
        cond = jnp.logical_and(iterations < 10, p_norm <1e-10)
        return cond

    def body(state):
        p, p_norm, i = state
        p, p_norm  = p_vec(i)
        return p, p_norm, i+1

    state = lax.while_loop(cond, body, init())
    p, p_norm, i = state
    return p / p_norm, i

new_p_vec_jit = jit(new_p_vec, static_argnums=(0,))


def new_r_vec(A, V, j,  gamma):
    """Generates a new r vector which is orthogonal to all previous V vectors
    """
    m, n = A.shape
    idx = jnp.arange(V.shape[1])
    j_mask =  idx < j

    def r_vec(i):
        r = random.uniform(KEYS[i], (m,))
        r = A.trans(r)
        r_norm = norm(r)
        r, r_norm, iters = reorth_mgs(V, r, r_norm, j_mask, gamma)
        return r, r_norm

    def init():
        r, r_norm = r_vec(0)
        return r, r_norm, 1

    def cond(state):
        r, r_norm, iterations = state
        cond = jnp.logical_and(iterations < 10, r_norm <1e-10)
        return cond

    def body(state):
        r, r_norm, i = state
        r, r_norm  = r_vec(i)
        return r, r_norm, i+1

    state = lax.while_loop(cond, body, init())
    r, r_norm, i = state
    return r / r_norm, i

new_r_vec_jit = jit(new_r_vec, static_argnums=(0,))