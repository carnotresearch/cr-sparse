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


from jax import jit, lax

import jax.numpy as jnp
from jax.numpy.linalg import qr, norm
from jax.scipy.linalg import svd

import cr.nimble as cnb

eps = jnp.finfo(float).eps

def proj_l2_ball(q=1., b=None, A=None):

    if b is not None:
        b = jnp.asarray(b)
        b = cnb.promote_arg_dtypes(b)

    if A is not None:
        A = jnp.asarray(A)
        A = cnb.promote_arg_dtypes(A)

    if q <= 0:
        raise ValueError("q must be greater than 0")

    @jit
    def proj_q(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        norm_x = norm(x)
        invalid =  norm_x > q

        def proj_inside(x):
            return (q/norm_x) * x

        return lax.cond(invalid,
            # scale down within the norm ball
            proj_inside,
            # keep as it is
            lambda x: x,
            x)

    if b is None and A is None:
        return proj_q


    @jit
    def proj_q_b(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        # compute difference from center
        r = x - b
        norm_r = norm(r)
        invalid =  norm_r > q

        def proj_inside(r):
            #TODO the adjustment term should depend on dtype
            factor = (q - 1e-7) * jnp.reciprocal(norm_r)
            return factor * r

        # update r if required
        r = lax.cond(invalid,
            # scale down within the norm ball
            proj_inside,
            # keep as it is
            lambda r: r,
            r)
        # move the vector to the center b
        x = r + b
        return x

    if A is None:
        return proj_q_b

    if b is None:
        # we have q and A specified. 
        # default value for b
        b = 0.

    raise NotImplementedError("|| A x -b || <= q is not supported yet.")
    # We know that A is specified
    U, S, Vh = svd(A, full_matrices=False)
    m, n = A.shape
    rnk = min(m,n)
    # square of singular values
    S2 = S**2

    # @jit
    def proj_q_b_A(x):
        # compute the residual vector
        r = A @ x - b
        invalid = norm(r) > q

        def project_inside(a):
            """Minimizes ||x - y|| s.t. || Ax - b || <= q

            TODO complete this
            """
            lambda0 = 0
            max_iters = 70
            tol = 1e-8*q

            # map b to the correct frame
            bb =  U.T @ b - S * (Vh @ a)
            print(bb)
            b2 = abs(bb)**2
            q2 = q**2

            l = lambda0
            oldff = 0
            one = jnp.ones(rnk)

            return x

        return project_inside(x)

        # update x if required
        x = lax.cond(invalid,
            project_inside,
            # keep as it is
            lambda x: x,
            x)
        return x
    
    return proj_q_b_A



def proj_l1_ball(q=1., b=None):
    r"""Projector functions for || x - b ||_1 <= q


    Algorithm 2 in Probing the Pareto frontier is a variant
    of this algorithm based on heap data structure and partial
    cumulative sums.
    """

    if b is not None:
        b = jnp.asarray(b)
        b = cnb.promote_arg_dtypes(b)

    # TODO: This creates problems with JIT
    # if q <= 0:
    #     raise ValueError("q must be greater than 0")


    def project_inside_ball(y):
        # sort the absolute values in descending order
        u = jnp.sort(jnp.abs(y))[::-1]
        # compute the cumulative sums
        cu = jnp.cumsum(u)
        # find the index where the cumulative sum is below the threshold
        cu_diff = cu - q
        u_scaled = u*jnp.arange(1, 1+len(u))
        flags = cu_diff > u_scaled
        K = jnp.argmax(flags)
        K = jnp.where(K == 0, len(flags), K)
        # compute the shrinkage threshold
        kappa = (cu[K-1] - q)/K
        # perform shrinkage
        return jnp.maximum(0, y - kappa) + jnp.minimum(0, y + kappa)




    @jit
    def proj_q(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        invalid = cnb.arr_l1norm(x) > q
        return lax.cond(invalid, 
            # find the shrinkage threshold and shrink
            lambda x: project_inside_ball(x),
            # no changes necessary
            lambda x : x, 
            x)

    if b is None:
        return proj_q


    @jit
    def proj_q_b(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        # compute difference from center
        r = x - b
        invalid = cnb.arr_l1norm(r) > q
        # update the residual
        r = lax.cond(invalid, 
            # find the shrinkage threshold and shrink
            lambda r: project_inside_ball(r),
            # no changes necessary
            lambda r : r, 
            r)
        # translate to the center
        return r + b

    return proj_q_b
