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
from jax.ops import index, index_add
from jax.numpy.linalg import norm
from .defs import SingleRecoverySolution
from cr.sparse import *

def solve_smv(dictionary, x, max_iters=None, max_res_norm=None):
    # initialize residual
    r = x
    num_atoms = dictionary.shape[0]
    z = jnp.zeros(num_atoms)
    # iteration count
    t = 0
    # compute the norm of original signal
    x_norm = norm(x)
    # absolute limit on res norm
    upper_res_norm = x_norm * 1e-6
    # upper limit on number of iterations
    upper_iters = 4 * num_atoms
    if max_iters is not None:
        upper_iters = max_iters
    if max_res_norm is not None:
        upper_res_norm = max_res_norm
    while True:
        # Compute the inner product of residual with atoms
        correlations = r @ dictionary.T
        # each correlation column is for one signal
        # take absolute values
        abs_corrs = jnp.abs(correlations)
        # find the maximum in the column
        best_match_index = jnp.argmax(abs_corrs)
        # pick corresponding correlation value
        coeff = correlations[best_match_index]
        # update the representation
        z  = index_add(z, index[best_match_index], coeff)
        # find the best match atom
        atom = dictionary[best_match_index]
        # update the residual
        r = r - coeff * atom
        t += 1
        # compute the updated residual norm
        r_norm = norm(r)
        # print("[{}] norm: {}".format(t, r_norm))
        # print('.', end="", flush=True)
        if t >= upper_iters:
            break
        if r_norm < upper_res_norm:
            break
        #print("[{}] res norm: {}".format(t, r_norm))
    solution = SingleRecoverySolution(signals=x, 
        representations=z, 
        residuals=r, 
        residual_norms=r_norm,
        iterations=t)
    return solution


def solve_mmv(dictionary, signals, max_iters=None, max_res_norm=None):
    # initialize residual
    residuals = signals
    num_signals = signals.shape[0]
    num_atoms = dictionary.shape[0]
    sol_shape = (num_signals, num_atoms)
    z = jnp.zeros(sol_shape)
    # iteration count
    t = 0
    # compute the norm of original signal
    x_norms = norms_l2_rw(signals)
    # absolute limit on res norm
    upper_res_norm = jnp.max(x_norms) * 1e-6
    # upper limit on number of iterations
    upper_iters = 4 * num_atoms
    while True:
        # Compute the inner product of residual with atoms
        correlations = jnp.matmul(dictionary, residuals.T)
        #print(correlations.shape)
        # each correlation column is for one signal
        # take absolute values
        abs_corrs = jnp.abs(correlations)
        # find the maximum in the column
        indices = jnp.argmax(abs_corrs, axis=0)
        for i in range(num_signals):
            # best match atom index
            best_match_index = indices[i]
            # pick corresponding correlation value
            coeff = correlations[best_match_index, i]
            # update the representation
            z  = index_add(z, index[i, best_match_index], coeff)
            # find the best match atom
            atom = dictionary[best_match_index]
            # update the residual
            update = coeff * atom
            residuals = index_add(residuals, index[i, :], -update)
        t += 1
        # compute the updated residual norm
        r_norms = norms_l2_rw(residuals)
        max_r_norm = jnp.max(r_norms)
        # print("[{}] norm: {}".format(t, r_norm))
        print('.', end="", flush=True)
        if ((max_iters is not None and t >= max_iters) or (max_res_norm is not None and max_r_norm < max_res_norm) or (max_r_norm < upper_res_norm) or (t >= upper_iters)):
            break
        #print("[{}] res norm: {}".format(t, max_r_norm))
    solution = SingleRecoverySolution(signals=signals, 
        representations=z, 
        residuals=residuals, 
        residual_norms=r_norms,
        iterations=t)
    return solution

