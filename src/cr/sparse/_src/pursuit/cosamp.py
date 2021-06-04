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


import jax.numpy as jnp
from jax import vmap


from .defs import RecoverySolution

from .util import largest_indices

EXTRA_FACTOR = 2

def solve(Phi, y, K, max_iters=None, res_norm_rtol=1e-3):
    """Solves the sparse recovery problem :math:`y = \Phi x + e` using Compressive Sampling Matching Pursuit
    """
    M, N = Phi.shape
    K2 = EXTRA_FACTOR * K
    K3 = K + K2
    # initialize residual
    r = y
    # Let's conduct first iteration of OMP
    # squared norm of the signal
    y_norm_sqr = y.T @ y
    # initialize residual squared norm with the same
    r_norm_sqr = y_norm_sqr
    # Number of iterations
    iterations = 0
    max_r_norm_sqr = y_norm_sqr * (res_norm_rtol ** 2) 
    I = jnp.array([])
    flags = jnp.zeros(N, dtype=bool)
    if max_iters is None:
        max_iters = M // 2
    for _ in range(max_iters):
        # compute the correlations
        h = Phi.T @ r
        I_2k = largest_indices(h, K2 if iterations else K3)
        # I_3k = jnp.union1d(I, I_2k).astype(int)
        flags = flags.at[I_2k].set(True)
        I_3k, = jnp.where(flags)
        Phi_3I = Phi[:, I_3k]
        # Solve least squares over the selected indices
        x_3I, r_3I_norms, rank_3I, s_3I = jnp.linalg.lstsq(Phi_3I, y)
        # pick the K largest indices
        Ia = largest_indices(x_3I, K)
        I = I_3k[Ia]
        x_I = x_3I[Ia]
        # Identify corresponding atoms
        Phi_I = Phi[:, I]
        # Compute new residual
        r = y - Phi_I @ x_I
        # Compute residual norm squared
        r_norm_sqr = r.T @ r
        iterations += 1
        # Check for convergence
        if r_norm_sqr < max_r_norm_sqr:
            break
        flags = flags.at[:].set(False)
        flags = flags.at[I].set(True)
    return RecoverySolution(x_I=x_I, I=I, r=r, r_norm_sqr=r_norm_sqr, iterations=iterations)
