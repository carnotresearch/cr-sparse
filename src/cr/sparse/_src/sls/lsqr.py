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

"""
LSQR algorithm for linear operators as defined by CR-Sparse

This implementation is adapted from PYLOPS.

* A is the operator for which we are solving A x = b
* alpha and beta are the terms emerging from lower bidiagonalization of A
* alpha is on the main diagonal.
* beta is on the lower diagonal.
* A V_k = U_{k+1} B
* rho and theta are the terms emerging from the upper bidiagonalization of A
* rho is on the main diagonal.
* theta is on the upper diagonal.
* A V_k = P_k R_k 


Relevant quantities for stopping criteria

- residual norm relative to norm of b
- A^H r norm relative to A^T b norm
- Condition number of A
- Norm of x
- Frobenius norm of A


Reference:

1982, Paige and Sauders,  LSQR: An algorithm for sparse linear 
equations and sparse least squares
"""
from typing import NamedTuple

import jax.numpy as jnp
from jax import lax, jit

#norm = jnp.linalg.norm

def l2norm(x):
    xh = jnp.conjugate(x)
    x_sqr = xh * x
    sum_sqr = jnp.sum(x_sqr)
    return jnp.sqrt(jnp.abs(sum_sqr))

def l2norm_sqr(x):
    xh = jnp.conjugate(x)
    x_sqr = xh * x
    return jnp.sum(x_sqr)


from cr.nimble import promote_arg_dtypes
from cr.sparse import RecoveryFullSolution

class LSQRState(NamedTuple):
    """State for LSQR algorithm
    """
    x: jnp.ndarray
    w: jnp.ndarray
    """The solution"""
    u: jnp.ndarray
    v: jnp.ndarray
    # various scalars to be retained. but they are stored as ndarray
    alpha: jnp.ndarray
    """Main diagonal term for lower bidiagonalization"""
    beta: jnp.ndarray
    """Lower diagonal term for lower bidiagonalization"""
    # recurrence relation 4.12
    rho_bar: jnp.ndarray
    phi_bar: jnp.ndarray
    # Rotation terms for upper to lower bidiagonalization of R
    # Section 5.2
    z: jnp.ndarray
    cs2 : jnp.ndarray
    sn2 : jnp.ndarray
    # Variables for tracking the norms
    D_norm_sqr: jnp.ndarray
    """Squared norm of D_k"""
    cum_z_sqr: jnp.ndarray
    cum_psi_sqr: jnp.ndarray
    # Stopping criteria related norm estimates
    A_norm: jnp.ndarray
    """Lower estimate of norm of A"""
    A_cond : jnp.ndarray
    """Lower estimate on the condition number of A"""
    x_norm: jnp.ndarray
    """Estimate on the norm of the solution vector x"""
    r_norm: jnp.ndarray
    """Estimate on the norm of the residual"""
    atr_norm: jnp.ndarray
    """Estimate on the norm of the proxy residual A^H r"""
    # Computation amount
    iterations: int
    """The number of iterations it took to complete"""
    n_times: int = 0
    """Number of times A x computed """
    n_trans : int = 0
    """Number of times A.T b computed """

    def __str__(self):
        """Returns the string representation of the algorithm state
        """
        s = []
        for x in [
            f'x: {self.x.shape}',
            f'w: {self.w.shape}',
            f'u: {self.u.shape}',
            f'v: {self.v.shape}',
            f'alpha: {self.alpha}',
            f'beta: {self.beta}',
            f'rho_bar: {self.rho_bar}',
            f'phi_bar: {self.phi_bar}',
            f'z: {self.z}',
            f'cs2: {self.cs2}',
            f'sn2: {self.sn2}',
            f'D_norm_sqr: {self.D_norm_sqr}',
            f'cum_z_sqr: {self.cum_z_sqr}',
            f'cum_psi_sqr: {self.cum_psi_sqr}',
            f'A_norm: {self.A_norm}',
            f'A_cond: {self.A_cond}',
            f'x_norm: {self.x_norm}',
            f'r_norm: {self.r_norm}',
            f'atr_norm: {self.atr_norm}',
            f'iterations: {self.iterations}',
            f'n_times: {self.n_times}',
            f'n_trans: {self.n_trans}',
            ]:
            s.append(x.rstrip())
        return u'\n'.join(s)

class LSQRSolution(NamedTuple):
    """Solution for LSQR algorithm
    """
    x: jnp.ndarray
    """ Solution vector """
    A_norm: jnp.ndarray
    """Lower estimate of norm of A"""
    A_cond : jnp.ndarray
    """Lower estimate on the condition number of A"""
    x_norm: jnp.ndarray
    """Estimate on the norm of the solution vector x"""
    r_norm: jnp.ndarray
    """Estimate on the norm of the residual"""
    atr_norm: jnp.ndarray
    """Estimate on the norm of the proxy residual A^H r"""
    iterations: int
    """The number of iterations it took to complete"""
    n_times: int = 0
    """Number of times A x computed """
    n_trans : int = 0
    """Number of times A.T b computed """

    def __str__(self):
        """Returns the string representation of the algorithm solution
        """
        s = []
        for x in [
            f'x: {self.x.shape}',
            f'A_norm: {self.A_norm}',
            f'A_cond: {self.A_cond}',
            f'x_norm: {self.x_norm}',
            f'r_norm: {self.r_norm}',
            f'atr_norm: {self.atr_norm}',
            f'iterations: {self.iterations}',
            f'n_times: {self.n_times}',
            f'n_trans: {self.n_trans}',
            ]:
            s.append(x.rstrip())
        return u'\n'.join(s)

def lsqr(A, b, x0, damp=0, atol=1e-8, btol=1e-8, conlim=100000000., max_iters=10):
    """Solves the overdetermined system :math:`A x = b` in least square sense using LSQR algorithm.

    Args:
        A (cr.sparse.lop.Operator): A linear operator
        b (jax.numpy.ndarray): The target values
        x0 (jax.numpy.ndarray): Initial estimate
        damp (float): Paramter for damped least square problem
        atol (float): Tolerance for stopping criteria S1 and S2
        btol (float): Tolerance for stopping criteria S1
        conlim (float): Maximum allowed limit for condition number
        max_iters (int): maximum number of LSQR iterations

    Returns:
        LSQRSolution: A named tuple consisting of LSQR solution

    Adapted from Pylops
    """
    # Make sure that everything is in floating point
    b, x0 = promote_arg_dtypes(b, x0)
    # tolerance on condition number
    ctol = 0 if conlim == 0 else 1. / conlim
    # square the value of damping parameter to be used later
    damp_sqr = damp ** 2
    # initial residual after compenstating for x0
    r = b - A.times(x0)
    # norm of the initial residual
    b_norm = l2norm(r)
    # Eq 3.1 Step 1
    # first u vector is based on the initial residual
    u = r
    # beta_1 is same as norm of the initial residual
    beta = b_norm
    # normalize u
    u = lax.cond(beta > 0, 
            lambda _: u / beta, 
            lambda _: u, operand=None)
    # compute first v Eq 3.1 Step 1
    v = A.trans(u)
    # compute alpha_1 by normalizing v
    alpha = l2norm(v)
    # normalize v
    v = lax.cond(alpha > 0, 
            lambda _: v / alpha, 
            lambda _: v, operand=None)
    # theta_1 is norm of A^T r Eq 3.12
    atr_norm0 = alpha * beta

    def init():
        """Initialize the state for the LSQR algorithm
        """
        # initialize other variables
        # Step 1 Eq 4.10 d = v/rho. w = rho d = v.
        w = v
        # Step 1 Eq 4.12 see initial value for recurrence
        rho_bar = alpha
        # Step 1 Eq 4.12 see initial value for recurrence
        phi_bar = beta
        # initialize the algorithm state
        # note that while we are keeping x_0 in the state
        # we are keeping w_1 (corresponding to d_1)
        # in the state
        # The state contains
        # x_0, w_1, u_1, v_1, alpha_1, beta_1,
        # rho_bar_1, phi_bar_1 
        state = LSQRState(x=x0, w=w, u=u, v=v,
            alpha=alpha, beta=beta, 
            rho_bar=rho_bar, phi_bar=phi_bar,
            z=0, cs2=-1, sn2=0,
            D_norm_sqr=0., cum_z_sqr=0., cum_psi_sqr = 0,
            A_norm=0., A_cond=1., 
            x_norm = l2norm(x0), r_norm=b_norm, atr_norm=atr_norm0,
            iterations=1, n_times=1, n_trans=1)
        return state

    def cond(state):
        """Checks the halting criteria. 

        Returns:
            bool: True if algorithm should continue, False otherwise

        We use the norms estimated in the body of the algorithm to compute
        quantities which can be checked for convergence.

        Refer to the stopping criteria S1,S2,S3 in the original paper.
        """
        # limit on number of iterations
        more_iters = state.iterations < max_iters
        # Compute the current residual norm relative to original residual normm S1
        rel_r_norm = state.r_norm / b_norm
        # Compute the current A^H r norm relative to original A^H r norm
        rel_atr_norm = state.atr_norm / atr_norm0
        # Compute the inverse of the condition number S3
        inv_cond_num = 1. / state.A_cond
        # combined tolerance for relative residual norm S1
        rtol = btol + atol * state.A_norm * state.x_norm / b_norm
        # A measure on the whether residual norm is below machine precision
        t1 = rel_r_norm / (1. + state.A_norm * state.x_norm / b_norm)

        # all the required stopping criteria

        # S3 inverse of condition number must be greater than the tolerance
        valid_inv_cond_num = inv_cond_num > ctol
        # Relative ATR norm must be above a threshold to continue
        rel_atr_high = rel_atr_norm > atol
        # Relative residual norm must be above a threshold to continue
        rel_r_norm_high = rel_r_norm > rtol
        # build up the combined condition
        condition = more_iters
        # check that none of the thresholds are too small
        # These guard against extremely small values [below machine precision]
        condition = jnp.logical_and(condition, 1 + inv_cond_num > 1)
        condition = jnp.logical_and(condition, 1 + rel_atr_norm > 1)
        condition = jnp.logical_and(condition, 1 + t1 > 1)
        # user defined thresholds
        condition = jnp.logical_and(condition, valid_inv_cond_num)
        condition = jnp.logical_and(condition, rel_atr_high)
        condition = jnp.logical_and(condition, rel_r_norm_high)
        # Return th combined condition
        return condition

    def body(state):
        """Main body of each iteration of LSQR algorithm
        """
        ## next round of biorthogonalization
        # update u step 3.a
        u = A.times(state.v) - state.alpha * state.u
        n_times = state.n_times + 1
        # update beta step 3.a 
        beta = l2norm(u)
        # normalize u step 3.a
        u = lax.cond(beta > 0, 
                lambda _: u / beta, 
                lambda _: u, operand=None)
        # update A_norm Near eq 5.10, expression for B_k Frobenius norm update
        A_norm = l2norm(jnp.array([state.A_norm, state.alpha, beta, damp]))
        # update v
        v = A.trans(u) - beta * state.v
        n_trans = state.n_trans + 1
        # compute alpha_1 by normalizing v
        alpha = l2norm(v)
        # normalize v
        v = lax.cond(alpha > 0, 
                lambda _: v / alpha, 
                lambda _: v, operand=None)
        
        ## plane rotation to eliminate the damping parameter
        # This code is a no-op if damp = 0
        rho_bar = l2norm(jnp.array([state.rho_bar, damp]))
        cs1 = state.rho_bar / rho_bar
        sn1 = damp / rho_bar
        psi = sn1 * state.phi_bar
        phi_bar = cs1 * state.phi_bar

        ## plane rotation to eliminate the subdiagonal element beta
        # recurrence relation Eq 4.12
        # step 4.a (solution of eq 4.12 cell 1,1 and cell 2,1 of R.H.S.)
        # rho(k) from rho_bar(k) and beta(k+1)
        rho = l2norm(jnp.array([rho_bar, beta]))
        # step 4.b
        # c(k) from rho_bar(k) and rho(k)
        cs = rho_bar / rho
        # step 4.c
        # s(k) from beta(k+1) and rho(k)
        sn = beta / rho
        # step 4.d (eq 4.12 cell 1,2 of RHS)
        # theta(k+1) from s(k) and alpha(k+1)
        theta = sn * alpha
        # step 4.e (eq 4.12 cell 2,2 of RHS)
        # rho_bar(k+1) from c(k) and alpha(k+1)
        rho_bar = -cs * alpha
        # step 4.f (eq 4.12 cell 1,3 of RHS)
        # phi(k) from c(k) and phi_bar(k)
        phi = cs * phi_bar
        # step 4.g (eq 4.12 cell 2,3 of RHS)
        # phi_bar(k+1) from s(k) and phi_bar(k)
        phi_bar = sn * phi_bar

        ## update x and w.
        # w = rho d   => d  = w/rho
        # step sizes have to be scaled accordingly by rho
        # step size for step 5.a  Eq 4.11
        t1 = phi / rho
        # step size for step 5.b Eq 4.10
        t2 = -theta / rho
        # Note: we need to update w first as per the paper
        # But it seems that the algorithm works correctly only if 
        # x is updated according to previous w
        # update x , step 5.a Eq 4.11
        x = state.x + t1 * state.w
        # update w, step 5.b Eq 4.10
        w = v + t2 * state.w
        # The column d_k for the matrix D_k
        dk = state.w / rho
        # Frobenius norm squared for the matrix D_k
        D_norm_sqr = state.D_norm_sqr + l2norm_sqr(dk)

        # use a plane rotation on the right to eliminate the
        # super-diagonal element (theta) of the upper-bidiagonal matrix.
        # Then use the result to estimate norm(x).
        delta = state.sn2 * rho
        gamma_bar = - state.cs2 * rho
        rhs = phi - delta * state.z
        z_bar = rhs / gamma_bar
        # Estimate of | x | 
        x_norm = jnp.sqrt(state.cum_z_sqr + z_bar**2)
        gamma = l2norm(jnp.array([gamma_bar, theta]))
        cs2 = gamma_bar / gamma
        sn2 = theta / gamma
        z = rhs / gamma
        cum_z_sqr = state.cum_z_sqr + z ** 2.

        # test for convergence. First, estimate the condition of the matrix
        # Opbar, and the norms of rbar and Opbar'rbar
        # Lower estimate of condition number of A Eq 5.10
        A_cond = A_norm * jnp.sqrt(D_norm_sqr)
        # Estimate of residual norm squared Eq 5.2 |r_k|^2
        res1 = phi_bar ** 2
        cum_psi_sqr = state.cum_psi_sqr + psi ** 2
        r_norm = jnp.sqrt(res1 + cum_psi_sqr)
        # Estimated norm of A^T r Eq 5.4 (with some change of variables)
        atr_norm = alpha * abs(sn * phi)

        # distinguish between r1norm = ||b - Ax|| and
        # r2norm = sqrt(r1norm^2 + damp^2*||x||^2).
        # Estimate r1norm = sqrt(r2norm^2 - damp^2*||x||^2).
        # Although there is cancellation, it might be accurate enough.
        r1sq = r_norm ** 2 - damp_sqr * cum_z_sqr
        r1norm = jnp.sign(r1sq) * jnp.sqrt(jnp.abs(r1sq))
        r2norm = r_norm

        # update state
        state = LSQRState(x=x, w=w, u=u, v=v,
            alpha=alpha, beta=beta,
            rho_bar=rho_bar, phi_bar=phi_bar,
            z=z, cs2=cs2, sn2=sn2,
            D_norm_sqr=D_norm_sqr, cum_z_sqr=cum_z_sqr, cum_psi_sqr=cum_psi_sqr,
            A_norm=A_norm, A_cond=A_cond, 
            x_norm=x_norm, r_norm=r_norm, atr_norm=atr_norm,
            iterations=state.iterations+1, n_times=n_times, n_trans=n_trans)
        return state

    # state = init()
    # # print(state)
    # while cond(state):
    #     state = body(state)
    #     # print(state)

    state = lax.while_loop(cond, body, init())

    # Solution
    return LSQRSolution(x=state.x, 
        A_norm=state.A_norm, A_cond=state.A_cond, 
        x_norm=state.x_norm, r_norm=state.r_norm, atr_norm=state.atr_norm,
        iterations=state.iterations, n_times=state.n_times, n_trans=state.n_trans)


lsqr_jit = jit(lsqr, static_argnums=(0, 3, 4, 5,6, 7))
