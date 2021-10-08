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

"""
Utility functions
"""

from jax import jit
import jax.numpy as jnp

import cr.sparse as crs

def sizes_from_labels(labels, k):
    """Returns the cluster sizes for each label
    """
    labels = labels[jnp.newaxis, :]
    clusters = jnp.arange(k)[:, jnp.newaxis]
    match = labels == clusters
    return jnp.sum(match, axis=1)

sizes_from_labels_jit = jit(sizes_from_labels, static_argnums=(1,))

def start_end_indices(cluster_sizes):
    """Returns the start and end indices for each cluster
    """
    cluster_sizes = jnp.asarray(cluster_sizes)
    k = len(cluster_sizes)
    start_indices = jnp.cumsum(cluster_sizes)
    start_indices = crs.vec_shift_right(start_indices)
    end_indices = start_indices + cluster_sizes
    return start_indices, end_indices


def labels_from_sizes(sizes):
    """Returns cluster labels from cluster sizes

    TODO: Not jittable
    """
    sizes = jnp.asarray(sizes)
    K = len(sizes)
    labels = [jnp.ones(size) * k for k, size in enumerate(sizes)]
    return jnp.concatenate(labels)