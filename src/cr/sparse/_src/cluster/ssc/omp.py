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

from jax import jit, vmap
import jax.numpy as jnp

import cr.sparse as crs
import cr.sparse.la as crla

submat_multiplier = vmap(crla.mult_with_submatrix, (None, 1, 1), 1)
submat_solver = vmap(crla.solve_on_submatrix, (None, 1, 1), (1, 1,))


def build_representation_omp(X, K):
    # Ambient dimension
    D = X.shape[0]
    # Number of data points
    N = X.shape[1]
    # We normalize the columns of X
    Xn = crs.normalize_l2_cw(X)
    # The dictionary
    Dict = Xn
    # The residual 
    R = Xn
    # Let's conduct first iteration of OMP
    # The proxy representation
    P = Dict.T @ R
    # First correlation of residual with signal
    # Set the diagonal to zero
    H = crs.set_diagonal(P, 0)
    # Index of best match
    indices = crs.abs_max_idx_cw(H)
    # Initialize the array of selected indices
    # with current indices as the first row
    I = indices[jnp.newaxis, :]
    Z, R = submat_solver(Dict, I, X)
    # conduct OMP iterations
    for k in range(1, K):
        # compute the correlations
        H = crs.set_diagonal(Dict.T @ R, 0)
        # Index of best match
        indices = crs.abs_max_idx_cw(H)
        # Update the set of indices
        I = jnp.vstack((I, indices))
        # Solve over these indices
        Z, R = submat_solver(Dict, I, X)
    return Z, I, R

build_representation_omp_jit = jit(build_representation_omp, static_argnums=(1,))

