# Copyright 2021 Carnot Research Pvt Ltd
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


import jax.numpy as jnp
from jax.ops import index, index_add, index_update
from jax.numpy.linalg import norm, lstsq
from .defs import SingleRecoverySolution

from .util import abs_max_idx, gram_chol_update
from cr.sparse.la import solve_spd_chol_solve

def solve(Phi, x, max_iters, max_res_norm=1e-6):
    # initialize residual
    r = x
    D = Phi.shape[0]
    N = Phi.shape[1]
    K = max_iters
    # Let's conduct first iteration of OMP
    # squared norm of the signal
    norm_x_sqr = x.T @ x
    # initialize residual squared norm with the same
    norm_r_sqr = norm_x_sqr
    # The proxy representation
    p = Phi.T @ x
    # First correlation of residual with signal
    h = p
    # Index of best match
    index = abs_max_idx(h)
    # Initialize the array of selected indices
    indices = jnp.array([index])
    # First matched atom
    atom = Phi[:, index]
    # Initial subdictionary of selected atoms
    subdict = jnp.expand_dims(atom, axis=1)
    # Initial L for Cholesky factorization of Gram matrix
    L = jnp.ones((1,1))
    # sub-vector of proxy corresponding to selected indices
    p_sub = p[indices]
    # sub-vector of representation coefficients estimated so far
    alpha_sub = p_sub
    # updated residual after first iteration
    r = x - subdict @ alpha_sub
    # norm squared of new residual
    norm_r_new_sqr = r.T @ r
    # conduct OMP iterations
    for k in range(1, K):
        norm_r_sqr = norm_r_new_sqr
        # compute the correlations
        h = Phi.T @ r
        # Index of best match
        index = abs_max_idx(h)
        # Update the set of indices
        indices = jnp.append(indices, index)
        # best matching atom
        atom = Phi[:, index]
        # Correlate with previously selected atoms
        b = subdict.T @ atom
        # Update the Cholesky factorization
        L = gram_chol_update(L, b)
        # Update the subdictionary
        subdict = jnp.hstack((subdict, jnp.expand_dims(atom,1)))
        # sub-vector of proxy corresponding to selected indices
        p_sub = p[indices]
        # sub-vector of representation coefficients estimated so far
        alpha_sub = solve_spd_chol_solve(L, p_sub)
        # updated residual after first iteration
        r = x - subdict @ alpha_sub
        # norm squared of new residual
        norm_r_new_sqr = r.T @ r
    return alpha_sub, indices, r
