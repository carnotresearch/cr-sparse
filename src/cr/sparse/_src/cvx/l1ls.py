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
Solves the l1 minimization problem using the Truncated Newton Interior Point Method


References: 

- An interior point method for large scale l-1 regularized least squares

Summary of variables involved in the algorithm

* x:  primal variable
* y: the observation vector/measurements y = A x + v
* v: the noise term
* A: the sensing matrix operator (Phi W)
* z: primal residual z = Ax - y
* lambda: regularization parameter
* lambda_i: used when regularization parameter is different for x_i
* Aty: A^T y 
* nu: dual variable/dual feasible point, eq 11, central path
* p_obj : primal objective, eq 3,6,7,8 
* d_obj : dual objective, eq 10
* s : scaling factor for constructing a dual feasible point from an arbitrary x, eq 11
* eta: duality gap, eq 12
* u : auxiliary primal variable, converting L1-LSP to QP, eq 13
* t : central path parameter
* mu: scale factor for t [2-50]
* t_0: initial value for the central path parameter, 1/lambda
* tol, epsilon: target duality gap
* 2n/t: how sub-optimal x(t) is
* H : Hessian of  Phi_t, eq 14
* g : gradient of Phi_t , eq 14
* s : step size for backtracking line search
* alpha, beta: parameters for backtracking line search
* s_min: parameter for t update
* d_1, d_2: diagonals for Hessian compact representation, IV.B
* g_1, g_2: parts of gradient of Phi_t
* g_1: gradient w.r.t. x
* g_2: gradient w.r.t. u
* P: the preconditioner, eq 15, 16
* tau: positive constant for the preconditioner, eq 16
* xi: parameter for PCG tolerance termination 

"""
from typing import NamedTuple, List, Dict

import jax.numpy as jnp
from jax import jit, lax
norm = jnp.linalg.norm


from cr.sparse.opt import pcg
from cr.sparse import RecoveryFullSolution

# IPM parameters
MAX_ITERS = 400

MU = 2

# PCG parameters
PCG_MAX_ITERS = 5000

# Line search parameters
ALPHA = 0.01
BETA = 0.5
# Maximum number of iterations for the backtracking line search
MAX_LS_ITER = 100


class State(NamedTuple):
    """State of the TNIPM algorithm
    """
    x: jnp.ndarray
    """Primal variable"""
    u: jnp.ndarray
    """Auxiliary Primal variable"""
    z: jnp.ndarray
    """Primal residual z = A x - y"""
    nu: jnp.ndarray
    """Dual variable"""
    dxu: jnp.ndarray
    """ Delta in x and u, the result of PCG step"""
    primal_obj: float
    "Primal objective function value"
    dual_obj: float
    "Dual objective function value"
    gap : float
    "duality gap"
    rel_gap : float
    "relative gap"
    s : float
    """ Step size for line search"""
    t : float 
    """ Central path parameter"""
    iterations: int
    """The number of iterations it took to complete"""
    n_times: int
    """Number of times A x computed """
    n_trans : int
    """Number of times A.T b computed """


def solve_from(A, y, lambda_, x0, u0, tol=1e-3, xi=1e-3, t0=None,
    max_iters=MAX_ITERS, pcg_max_iters=PCG_MAX_ITERS):
    """
    Solves :math:`\min \| A x - b \|_2^2 + \\lambda \| x \|_1` using the Truncated Newton Interior Point Method
    """
    trans = A.trans
    times = A.times
    #TODO check for zero solution
    Aty = trans(y)
    m = y.shape[0]
    n = Aty.shape[0]
    lambda_max = norm(Aty, jnp.inf)
    # initialize other parameters
    t0 = t0 if t0 is not None else jnp.minimum(jnp.maximum(1,1/lambda_),2*n/1e-3)
    # if lambda_ > lambda_max: we have a zero solution
    
    # Diagonal for 2 * A^T A (preconditioner simplified)
    diag_AtA = 2 * jnp.ones(n)

    def get_primal_obj(x, z):
        """Computes the primal objective from primal variables"""
        # eq 7
        return (jnp.vdot(z, z) + lambda_*norm(x,1))

    def get_dual_obj(nu):
        """Computes dual objective from dual variables"""
        # Eq 10
        return (-0.25* jnp.vdot(nu, nu) - jnp.vdot(nu,y))

    def get_nu(z):
        """Computes the dual variable nu from primal residual z"""
        nu = 2 * z # eq 11
        Atnu = trans(nu)
        Atnu_max  = norm(Atnu, jnp.inf)
        sf = lambda_ / Atnu_max  # eq 11
        # contract nu if necessary
        nu =  jnp.where(sf < 1, sf * nu, nu) # eq 11 
        return nu

    def get_phi(u, z, f, t):
        """
        Computes the log barrier Phi_t Sec IV.A [scaled by 1/t]
        """
        return jnp.vdot(z,z) + lambda_* jnp.sum(u) -jnp.sum(jnp.log(-f))/t

    def init():
        z = times(x0) - y
        nu = get_nu(z)
        # initial value of primal objective
        primal_obj  =  get_primal_obj(x0, z)
        # initial value of the dual objective
        dual_obj = get_dual_obj(nu)
        gap = primal_obj - dual_obj
        rel_gap = gap / dual_obj
        dxu = jnp.zeros(2*n)
        return State(x=x0, u=u0, z=z, nu=nu, dxu=dxu,
            primal_obj=primal_obj, dual_obj=dual_obj, gap=gap, rel_gap=rel_gap, 
            s=jnp.inf, t=t0,
            iterations=1, n_times=1, n_trans=2)

    def body(state):
        x = state.x
        u = state.u
        z = state.z
        nu = state.nu
        dxu = state.dxu
        t = state.t
        # print(f'{x=}')
        # print(f'{u=}')
        # print(f'{z=}')
        # print(f'{nu=}')
        # print(f'{dxu=}')
        # count of A.times in this iteration 
        n_times = 0
        # count of A.trans in this iteration
        n_trans = 0
        #--------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------
        # Newton step calculation
        #--------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------
        q1 = 1/(u+x)
        q2 = 1/(u-x)
        # note d1, d2 are scaled by 1/t w.r.t. D1, D2 in the paper
        d1 = (q1**2+q2**2)/t
        d2 = (q1**2-q2**2)/t

        # calculate gradient Sec IV.B
        upper = trans(2*state.z)-(q1-q2)/t 
        n_trans += 1
        lower = lambda_*jnp.ones(n)-(q1+q2)/t
        gradient_phi = jnp.concatenate((upper, lower))

        #--------------------------------------------------------------------------------
        # Hessian operator IV.B
        #--------------------------------------------------------------------------------
        def hessian(x):
            """Computes the Hessian of Phi for a given x : y = H x"""
            # split x into upper and lower parts
            x1 = x[:n]
            x2 = x[n:]
            upper = trans(2*times(x1)) + d1 * x1 + d2 * x2
            lower = d2 * x1 + d1 * x2
            y = jnp.concatenate((upper, lower))
            return y

        #--------------------------------------------------------------------------------
        # Preconditioner eq 15-16
        #--------------------------------------------------------------------------------
        # calculate vectors to be used in the preconditioner
        # eq 15-16
        # 2 diag(A^T A) + D_1/t
        prb = diag_AtA+d1
        # (D_1 D_3 - D_2^2)
        prs = prb*d1-(d2**2)
        p1 = d1 / prs
        p2 = d2 / prs
        p3 = prb / prs

        def preconditioner(x):
            """Computes the inverse y = M \ x where M is the preconditioner operator"""
            x1 = x[:n]
            x2 = x[n:]
            upper =  p1 * x1 - p2 * x2
            lower = -p2 * x1 + p3 * x2
            y = jnp.concatenate((upper, lower))
            return y

        #--------------------------------------------------------------------------------
        # Preconditioned Conjugate Gradients TNIPM step 1
        #--------------------------------------------------------------------------------
        # set pcg tol (relative)
        gradient_norm   = norm(gradient_phi)
        # See truncation rule
        eta = state.gap
        pcg_tol  = jnp.minimum(1e-1,xi*eta/jnp.minimum(1,gradient_norm))
        # pcg_tol = jnp.where (ntiter != 0 and pitr == 0, pcg_tol*0.1, pcg_tol)
        pcg_sol = pcg.solve_from(hessian, -gradient_phi, dxu, 
            max_iters=pcg_max_iters, tol=pcg_tol,
            M=preconditioner)
        dxu = pcg_sol.x
        dx = dxu[:n]
        du = dxu[n:]
        # how many iterations in PCG
        pcg_iters = pcg_sol.iterations
        # print(f"pcg: {pcg_tol=:.4f} {pcg_iters=}")
        # print(f'{dxu=}')
        # Every pcg iteration is one H(x) and one M(x)
        # M(x) is vector-vector stuff
        # H(x) involves one A x and one A^T x
        n_times += pcg_iters
        n_trans += pcg_iters
        #--------------------------------------------------------------------------------
        # Backtracking line search TNIPM step 2
        #--------------------------------------------------------------------------------
        f = jnp.concatenate((x-u, -x-u))
        phi = get_phi(u, z, f, t)
        gdx = jnp.vdot(gradient_phi, dxu)
        # print(f'{phi=}')
        # print(f'{gdx=}')
        def f_init(s):
            newx = x + s * dx
            newu = u + s * du
            newf = jnp.concatenate((newx-newu, -newx-newu))
            return (newx, newu, newf, s)

        def f_body(state):
            s = BETA * state[3]
            newx = x + s * dx
            newu = u + s * du
            newf = jnp.concatenate((newx-newu, -newx-newu))
            return (newx, newu, newf, s)

        def f_cond(state):
            newf = state[2]
            return jnp.max(newf) >= 0


        def bt_init(s):
            newx, newu, newf, s = lax.while_loop(f_cond, f_body, f_init(s))
            newz = times(newx) - y
            newphi =  get_phi(newu, newz, newf, t)
            times_count = 1
            return newx, newu, newf, newz, newphi, s, times_count

        def bt_body(state):
            newx, newu, newf, newz, newphi, s, times_count = state
            s = BETA * s
            newx, newu, newf, s = lax.while_loop(f_cond, f_body, f_init(s))
            newz = times(newx) - y
            newphi =  get_phi(newu, newz, newf, t)
            return newx, newu, newf, newz, newphi, s, times_count + 1

        def bt_cond(state):
            newphi = state[4]
            s = state[5]
            return newphi - phi > ALPHA * s * gdx

        newx, newu, newf, newz, newphi, s, times_count = lax.while_loop(bt_cond, bt_body, bt_init(1.0))
        # add the number of times A x was run in backtracking
        n_times += times_count
        #--------------------------------------------------------------------------------
        # x,u update TNIPM step 3
        #--------------------------------------------------------------------------------
        #update x
        x = newx
        # update u
        u = newu
        # update z (eq 9)
        z = newz
        #--------------------------------------------------------------------------------
        # Dual feasible point TNIPM step 4
        #--------------------------------------------------------------------------------
        # update nu (eq 11)
        nu = get_nu(z)
        #--------------------------------------------------------------------------------
        # Duality gap calculation TNIPM step 5
        #--------------------------------------------------------------------------------
        # update primal objective
        primal_obj  =  get_primal_obj(x, z)
        # update dual objective (only if it increases)
        dual_obj = jnp.maximum(get_dual_obj(nu), state.dual_obj)
        # duality gap
        gap = primal_obj - dual_obj
        # relative gap
        rel_gap = gap / state.dual_obj
        #--------------------------------------------------------------------------------
        # update t if required TNIPM step 7
        #--------------------------------------------------------------------------------
        # t = t if s < 0.5 else jnp.maximum(jnp.minimum(2*n*MU/gap, MU*t), t)
        t = jnp.where(s < 0.5, t, jnp.maximum(jnp.minimum(2*n*MU/gap, MU*t), t)) 
        return State(x=x, u=u, z=z, nu=nu, dxu=dxu,
            primal_obj=primal_obj, dual_obj=dual_obj, gap=gap, rel_gap=rel_gap,
            iterations=state.iterations+1, 
            n_times=state.n_times+n_times, 
            n_trans=state.n_trans+n_trans, t=t, s=s)

    def cond(state):
        """Condition for continuing the iterations TNIPM step 6
        """
        #print(f'[{state.iterations}] primal:{float(state.primal_obj):.3e} dual:{float(state.dual_obj):.3e} gap:{float(state.gap):.3e} rel:{float(state.rel_gap):.2f}')
        return (state.rel_gap > tol) & (state.iterations < max_iters)

    state = lax.while_loop(cond, body, init())

    # state = init()
    # while cond(state):
    #     state = body(state)
    return RecoveryFullSolution(x=state.x, r=-state.z,
        iterations=state.iterations, 
        n_times=state.n_times, n_trans=state.n_trans)

solve_from_jit  = jit(solve_from,
    static_argnames=("A", "tol", "xi", "t0", "max_iters", "pcg_max_iters"))

def solve(A, y, lambda_, x0=None, u0=None, tol=1e-3, xi=1e-3, t0=None,
    max_iters=MAX_ITERS, pcg_max_iters=PCG_MAX_ITERS):
    """
    Solves :math:`\min \| A x - b \|_2^2 + \\lambda \| x \|_1` using the Truncated Newton Interior Point Method
    """
    m, n = A.shape
    x0 = x0 if x0 is not None else jnp.zeros(n)
    u0 = u0 if u0 is not None else jnp.ones(n)
    return solve_from(A, y, lambda_, x0, u0, tol, xi, t0, max_iters, pcg_max_iters)

solve_jit  = jit(solve,
    static_argnames=("A", "x0", "u0", "tol", "xi", "t0", "max_iters", "pcg_max_iters"))
