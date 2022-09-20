# Copyright 2022 CR-Suite Development Team
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
Grassmannian frames
"""
import math

import jax.numpy as jnp
from jax import random, lax, jit

import cr.nimble as crn



def minimum_coherence(m, n):
    """Minimum achievable coherence for a Grassmannian frame
    """
    numer = n -m
    denom = m * (n - 1)
    return math.sqrt(numer / denom)

def build_grassmannian_frame(init, 
    frac=0.9, shrink_mu=0.9, iterations=25):
    """Builds a Grassmannian frame starting from a random matrix

    Args:
        init (jax.numpy.ndarray): initial frame
        frac (float): Threshold for fraction of off diagonal entries to keep/change
        shrink_mu (float): Factor by which to shrink or expand off diagonal entries
        iterations (int): Number of iterations for alternate projections

    Returns:
        (jax.numpy.ndarray) A frame which is approximately Grassmannian 

    It uses an alternate projections based algorithm.
    """
    m, n = init.shape
    # initial gram matrix
    g = init.T @ init

    # number of off diagonal entries in the Gram matrix
    n_off = n**2 - n
    upper_ind = round(frac * n_off) - 1
    lower_ind = round((1-frac)*n_off) - 1

    # indices for off diagonal entries
    off_ind = jnp.abs(g.ravel()-1) > 1e-6

    #@jit
    def body_fun(k, g):
        # flatten the gram matrix
        gg = g.ravel()
        # Absolute values of gram matrix
        abs_g = jnp.abs(gg)
        # sort the inner products by their absolute values
        sorted_g = jnp.sort(abs_g)

        ## Shrink the high inner products        
        # identify coherence values above the threshold
        upper_th = sorted_g[upper_ind]
        above_indices = abs_g > upper_th
        above_indices = jnp.logical_and(off_ind, above_indices)
        # Update off diagonal entries
        gg = jnp.where(above_indices, gg * shrink_mu, gg)

        ## Expand the near zero products
        lower_th = sorted_g[lower_ind]
        below_indices = abs_g < lower_th
        gg = jnp.where(below_indices, gg / shrink_mu, gg)

        # make the new gram matrix
        g = jnp.reshape(gg, (n, n))

        ## Reduce the rank of g back to m
        # perform SVD
        U, s, VT = jnp.linalg.svd(g)
        # Ensure that all higher singular values are set to 0
        s = s.at[m:].set(0)
        # Reconstruct the Gram matrix
        g = jnp.dot(U * s, VT)

        # Ensure that the diagonal elements of G continue to be 1
        d = jnp.diag(g)
        d2 = 1. / jnp.sqrt(d)
        g = d2 * g * d2
        return g

    # run the alternate projections
    g = lax.fori_loop(0, iterations, body_fun, g)
    # final dictionary
    U, s, VT = jnp.linalg.svd(g)
    s = s[:m]
    frame = crn.diag_premultiply(s ** 0.5, U[:, :m].T) 
    return frame

