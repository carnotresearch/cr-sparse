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

from typing import NamedTuple

import numpy as np
from scipy.optimize import linear_sum_assignment


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
    labels = [jnp.ones(size, dtype=int) * k for k, size in enumerate(sizes)]
    return jnp.concatenate(labels)


def best_map(true_labels, pred_labels):
    """Estimates the mapping  between true and estimated labels using Hungarian assignment
    """
    if len(true_labels) != len(pred_labels):
        raise ValueError("Number of labels must be same.")
    # identify the unique labels
    labels_1 = jnp.unique(true_labels)
    labels_2 = jnp.unique(pred_labels)
    # number of labels in set 1
    n_l1 = len(labels_1)
    # number of labels in set 2
    n_l2 = len(labels_2)
    # maximum number of labels (from both sets)
    n_labels = max(n_l1, n_l2)
    # prepare the cost matrix
    G = np.zeros((n_labels, n_labels))
    for i in range(n_l1):
        l1 = labels_1[i]
        ilabels = true_labels == l1
        for j in range(n_l2):
            l2 = labels_2[j]
            jlabels = pred_labels == l2
            G[i, j] = np.sum(ilabels & jlabels)
    # Run the Hungarian assignment algorithm
    rows, cols = linear_sum_assignment(-G)
    mapped_labels = jnp.zeros_like(pred_labels)
    for i in range(n_l2):
        old_label = labels_2[cols[i]]
        new_label = labels_1[i]
        mapped_labels = mapped_labels.at[pred_labels == old_label].set(new_label)
    return mapped_labels, cols, G 


class ClusteringError(NamedTuple):
    """Error between true labels and predicted labels
    """
    true_labels: jnp.ndarray
    """True labels assigned to each data point"""
    pred_labels: jnp.ndarray
    """Predicted labels assigned to each data point"""
    mapped_labels: jnp.ndarray
    """Predicted labels after relabeling by assignment method"""
    num_missed: int
    """Number of points where the true labels and predicted labels don't match"""
    error: float
    """Relative error (0 to 1)"""
    error_perc: float
    """Relative error in percentage"""


def clustering_error(true_labels, pred_labels):
    """Computes the clustering error between true labels and predicted labels
    """
    num_labels = len(true_labels)
    mapped_labels, mapping, _ = best_map(true_labels, pred_labels)
    num_missed = jnp.sum(true_labels != mapped_labels)
    error = num_missed / num_labels
    error_perc = error * 100
    return ClusteringError(true_labels=true_labels, pred_labels=pred_labels,
        mapped_labels=mapped_labels, num_missed=num_missed, 
        error=error, error_perc=error_perc)