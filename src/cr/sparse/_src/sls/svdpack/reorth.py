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

from cr.sparse import promote_arg_dtypes

def reorth_mgs(Q, r, r_norm, indices, alpha=0.5):
    """Reorthogonalizes r against subset of columns in Q indexed by indices

    If norm of r reduces significantly, then a second reorthogonalization is performed.
    If the norm of r reduces again significantly, then it is assumed that r is 
    numerically in the column span of Q and a zero vector is returned.
    """
    n, k = Q.shape
    k2 = len(indices)
    assert k2 <= k
    Q, r, r_norm = promote_arg_dtypes(Q, r, r_norm)

    def orthogonalize_with_col(i, r):
        # pick the corresponding column
        q = Q[:, i]
        # compute the dot product
        t = jnp.dot(q, r)
        # subtract the projection from r
        r = r - t * q
        return r

    def for_body(i, r):
        # orthogonalize against a column only if it is selected
        return lax.cond(indices[i], 
            lambda r : orthogonalize_with_col(i, r),
            lambda r : r,
            r
        )
    # orthogonalize r against Q
    r = lax.fori_loop(0, k2, for_body, r)
    old_norm = r_norm
    r_norm = norm(r)

    def while_cond(state):
        r, r_norm, old_norm, iterations = state
        return r_norm < alpha * old_norm

    def while_body(state):
        r, r_norm, old_norm, iterations = state
        # orthogonalize r against Q
        r = lax.fori_loop(0, k2, for_body, r)
        old_norm = r_norm
        r_norm = norm(r)
        return r, r_norm, old_norm, iterations+1

    state = r, r_norm, old_norm, 1
    state = lax.while_loop(while_cond, while_body, state)
    r, r_norm, old_norm, iterations = state
    return r, r_norm, iterations

reorth_mgs_jit = jit(reorth_mgs)

def reorth_noop(r, r_norm):
    return r, r_norm, 0
