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

from jax import jit, vmap
import jax.numpy as jnp

import cr.nimble as cnb

submat_multiplier = vmap(cnb.mult_with_submatrix, (None, 1, 1), 1)
submat_solver = vmap(cnb.solve_on_submatrix, (None, 1, 1), (1, 1,))


def build_representation_omp(X, K):
    """Builds K-sparse self-expressive representations of vectors in X in terms of other vectors in X
    """
    # Ambient dimension
    D = X.shape[0]
    # Number of data points
    N = X.shape[1]
    # We normalize the columns of X
    Xn = cnb.normalize_l2_cw(X)
    # The dictionary
    Dict = Xn
    # The residual 
    R = Xn
    # Let's conduct first iteration of OMP
    # The proxy representation
    P = Dict.T @ R
    # First correlation of residual with signal
    # Set the diagonal to zero
    H = cnb.set_diagonal(P, 0)
    # Index of best match
    indices = cnb.abs_max_idx_cw(H)
    # Initialize the array of selected indices
    # with current indices as the first row
    I = indices[jnp.newaxis, :]
    Z, R = submat_solver(Dict, I, X)
    # conduct OMP iterations
    for k in range(1, K):
        # compute the correlations
        H = cnb.set_diagonal(Dict.T @ R, 0)
        # Index of best match
        indices = cnb.abs_max_idx_cw(H)
        # Update the set of indices
        I = jnp.vstack((I, indices))
        # Solve over these indices
        Z, R = submat_solver(Dict, I, X)
    return Z, I, R

build_representation_omp_jit = jit(build_representation_omp, static_argnums=(1,))


def batch_build_representation_omp(X, K, batch_size):
    """Builds K-sparse self-expressive representations of vectors in X in terms of other vectors in X
    """
    # Ambient dimension
    D = X.shape[0]
    # Number of data points
    N = X.shape[1]
    n_batches =  (N + batch_size - 1) // batch_size
    starts = jnp.arange(n_batches) * batch_size
    ends = starts + batch_size
    starts = [i*batch_size for i in range(n_batches)]
    ends = [start+batch_size for start in starts]
    # last batch end would be different
    ends[-1] = N

    # We normalize the columns of X
    Xn = cnb.normalize_l2_cw(X)
    # The dictionary
    Dict = Xn
    Z_res = jnp.empty((K, N))
    I_res = jnp.empty((K, N), dtype=int)
    R_res = jnp.empty((D, N))
    # diagonal indices
    rr, c = jnp.diag_indices(batch_size)
    for batch in range(n_batches):
        start = starts[batch]
        end = ends[batch]
        # number of signals in the batch
        n2 = end - start
        # rows for setting the self inner product to 0.
        r = rr + start
        # Let's restrict our attention to this batch only
        X_batch = Xn[:, start:end]
        # The residual 
        R = X_batch
        # Let's conduct first iteration of OMP
        # The proxy representation
        P = Dict.T @ R
        # First correlation of residual with signal
        # Set the diagonal to zero
        H = P.at[(r, c)].set(0)
        # Index of best match
        indices = cnb.abs_max_idx_cw(H)
        # Initialize the array of selected indices
        # with current indices as the first row
        I_res = I_res.at[0, start:end].set(indices)
        Z, R = submat_solver(Dict, I_res[:1, start:end], X_batch)
        # conduct OMP iterations
        for k in range(1, K):
            # compute the correlations
            H = Dict.T @ R
            # Set the diagonal to zero
            H = H.at[(r, c)].set(0)
            # Index of best match
            indices = cnb.abs_max_idx_cw(H)
            # Update the set of indices
            I_res = I_res.at[k, start:end].set(indices)
            # Solve over these indices
            Z, R = submat_solver(Dict, I_res[:k+1, start:end], X_batch)
        Z_res = Z_res.at[:, start:end].set(Z)
        R_res = R_res.at[:, start:end].set(R)
    return Z_res, I_res, R_res

batch_build_representation_omp_jit = jit(batch_build_representation_omp, static_argnums=(1,2))