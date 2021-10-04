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

"""Similarity measures
"""

from jax import jit
import jax.numpy as jnp

def dist_to_gaussian_sim(dist, sigma):
    """Computes the Gaussian similarities for given distances
    """
    d = dist**2 / (2 * sigma**2)
    return jnp.exp(-d)

def sqr_dist_to_gaussian_sim(sqr_dist, sigma):
    """Computes the Gaussian similarities for given squared distances
    """
    d = sqr_dist / (2 * sigma**2)
    return jnp.exp(-d)

def eps_neighborhood_sim(dist, threshold):
    """Computes the epsilon neighborhood similarity
    """
    return dist < threshold