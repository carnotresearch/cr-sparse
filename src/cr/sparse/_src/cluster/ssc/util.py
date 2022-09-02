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

"""
Sparse Subspace Clustering functions
"""

from typing import NamedTuple

from jax import jit, vmap
import jax.numpy as jnp

import cr.nimble as cnb
import cr.sparse.cluster as crcluster
from jax.experimental.sparse import BCOO, sparsify


@jit
def sparse_to_full_rep(X, I):
    """Combines values and indices arrays to sparse representations
    """
    # number of signals
    n  = X.shape[1]
    mapper = lambda x, i : jnp.zeros(n).at[i].set(x)
    return vmap(mapper, (1,1), 1)(X, I)

@jit
def sparse_to_bcoo(X, I):
    """"Combines values and indices arrays to a BCOO formatted sparse matrix
    """
    # number of signals
    n  = X.shape[1]
    # output shape
    shape = (n, n)
    # sparsity level
    k = I.shape[0]
    # column numbers of each entry
    cols = jnp.arange(n)
    # repeat column numbers for k rows
    cols = jnp.tile(cols, (k,1))
    # total number of non-zero values
    nse = k*n
    # flatten rows and cols matrices
    cols = jnp.reshape(cols, (nse,1))
    rows = jnp.reshape(I, (nse, 1))
    # prepare combined indices list
    indices = jnp.hstack((rows, cols))
    # flatten values list
    values = jnp.reshape(X, nse)
    # combine values and indices
    Y = BCOO((values, indices), shape=(n,n))
    return Y

def bcoo_to_sparse(C, k):
    """Converts it back to values and indices (column-wise) format
    """
    rows, cols = C.indices.T
    data = C.data
    # number of values
    nse = len(data)
    # number of signals
    n = nse // k
    X = jnp.reshape(data, (k, n))
    I = jnp.reshape(rows, (k, n))
    return X, I

bcoo_to_sparse_jit = jit(bcoo_to_sparse, static_argnums=(1,))



@sparsify
def rep_to_affinity(Z):
    """Converts sparse representations to symmetric affinity matrix
    """
    Z = jnp.abs(Z)
    affinity = Z + Z.T
    return affinity

def angles_between_points(X):
    """Returns an SxS matrix of angles between each pair of points
    """
    # make sure that the points are normalized
    X = cnb.normalize_l2_cw(X)
    # Compute gram matrix
    G = X.T @ X
    # Avoid overflow in gram matrix
    G = jnp.minimum(G, 1)
    return jnp.rad2deg(jnp.arccos(G))


def min_angles_inside_cluster(angles, cluster_sizes):
    """Returns the minimum angles for for each point with its neighbors inside the cluster 
    """
    # we have to ignore the diagonal elements
    angles = cnb.set_diagonal(angles, 10000)
    start_indices, end_indices = crcluster.start_end_indices(cluster_sizes)
    K = len(cluster_sizes)
    def min_angles(k):
        start = start_indices[k]
        end = end_indices[k]
        A = angles[start:end, start:end]
        return jnp.min(A, axis=0)

    mins = [min_angles(k) for k in range(K)]
    return jnp.concatenate(mins)

def min_angles_outside_cluster(angles, cluster_sizes):
    """Returns the minimum angles for each point with its neighbors from all other clusters 
    """
    start_indices, end_indices = crcluster.start_end_indices(cluster_sizes)
    K = len(cluster_sizes)
    def min_angles(k):
        start = start_indices[k]
        end = end_indices[k]
        # pick the relevant rows
        A = angles[start:end, :]
        # set the angles inside the cluster to high value
        A = A.at[:, start:end].set(10000)
        # minimize on each row
        return jnp.min(A, axis=1)

    mins = [min_angles(k) for k in range(K)]
    return jnp.concatenate(mins)

def nearest_neighbors_inside_cluster(angles, cluster_sizes):
    """Returns the index of the nearest neighbor for each point inside the cluster 
    """
    # we have to ignore the diagonal elements
    angles = cnb.set_diagonal(angles, 10000)
    start_indices, end_indices = crcluster.start_end_indices(cluster_sizes)
    K = len(cluster_sizes)
    def inn_indices(k):
        start = start_indices[k]
        end = end_indices[k]
        A = angles[start:end, start:end]
        return jnp.argmin(A, axis=0) + start

    mins = [inn_indices(k) for k in range(K)]
    return jnp.concatenate(mins)

def nearest_neighbors_outside_cluster(angles, cluster_sizes):
    """Returns index of the nearest neighbor for each point with its neighbors from all other clusters 
    """
    start_indices, end_indices = crcluster.start_end_indices(cluster_sizes)
    K = len(cluster_sizes)
    def onn_indices(k):
        start = start_indices[k]
        end = end_indices[k]
        # pick the relevant rows
        A = angles[start:end, :]
        # set the angles inside the cluster to high value
        A = A.at[:, start:end].set(10000)
        # minimize on each row
        return jnp.argmin(A, axis=1)

    mins = [onn_indices(k) for k in range(K)]
    return jnp.concatenate(mins)


def sorted_neighbors(angles):
    """Returns the neighbor indices sorted by angle between points
    """
    # sort the angle row-wise (along the column axis)
    indices = jnp.argsort(angles)
    # drop the first column
    indices = indices[:, 1:]
    return indices


def inn_positions(labels, sorted_neighbor_labels):
    """Returns the position of a neighbor inside the cluster for each point in 
    its list of sorted neighbors across all clusters
    """
    inn_pos = lambda s: jnp.argmax(sorted_neighbor_labels[s, :] == labels[s])
    return vmap(inn_pos)(jnp.arange(labels.shape[0]))


class SubspacePreservationStats(NamedTuple):
    """Statistics for subspace preserving representations
    """
    spr_errors : jnp.ndarray
    spr_flags : jnp.ndarray
    spr_error : float
    spr_flag : bool
    spr_perc : float

    def __str__(self):
        s = []
        s.append(f'spr_error: {self.spr_error}, spr_flag: {self.spr_flag}, spr_perc: {self.spr_perc}')
        return '\n'.join(s)

def subspace_preservation_stats(C, labels):
    """Returns the statistics for subspace preservation
    """
    m, n = C.shape
    assert m == n, "C must be a square representation matrix"
    # we are concerned only with absolute values
    C = jnp.abs(C)

    def stats(i):
        # pick the i-th signal
        ci = C[:, i]
        # identify its cluster number
        k = labels[i]
        # identify non-zero entries
        non_zero_indices = ci >= 1e-3
        # identify the clusters of corresponding vectors
        non_zero_labels = jnp.where(non_zero_indices, labels, k)
        # verify that they all belong to same subspace
        spr_flag = jnp.all(non_zero_labels == k)
        # flags for current subspace
        w = labels == k
        # identify entries in current subspace
        cik = jnp.where(w, ci, 0)
        spr_error = 1 - jnp.sum(cik) / jnp.sum (ci)
        return spr_flag, spr_error

    spr_flags, spr_errors = vmap(stats)(jnp.arange(m))
    spr_error = jnp.mean(spr_errors)
    spr_flag = jnp.all(spr_flags)
    spr_perc = jnp.sum(spr_flags) * 100. / m
    return SubspacePreservationStats(spr_errors=spr_errors,
        spr_flags=spr_flags, spr_error=spr_error,
        spr_flag=spr_flag, spr_perc=spr_perc)

subspace_preservation_stats_jit = jit(subspace_preservation_stats)




def sparse_subspace_preservation_stats(Z, I, labels):
    """Returns the statistics for subspace preservation from sparse representations
    """
    # subpsace dimension and number of signals
    d, n = Z.shape
    # we are concerned only with absolute values
    Z = jnp.abs(Z)

    def stats(i):
        # pick the i-th signal
        ci = Z[:, i]
        # corresponding indices
        indices = I[:, i]
        # identify its cluster number
        k = labels[i]
        # identify the clusters of corresponding vectors
        non_zero_labels = labels[indices]
        # mark the labels for small coefficients to k
        non_zero_labels = jnp.where(ci < 1e-3, k, non_zero_labels)
        # verify that they all belong to same subspace
        spr_flag = jnp.all(non_zero_labels == k)
        # flags for current subspace
        w = labels == k
        # identify entries in current subspace
        cik = jnp.where(non_zero_labels == k, ci, 0)
        spr_error = 1 - jnp.sum(cik) / jnp.sum (ci)
        return spr_flag, spr_error

    spr_flags, spr_errors = vmap(stats)(jnp.arange(n))
    spr_error = jnp.mean(spr_errors)
    spr_flag = jnp.all(spr_flags)
    spr_perc = jnp.sum(spr_flags) * 100. / n
    return SubspacePreservationStats(spr_errors=spr_errors,
        spr_flags=spr_flags, spr_error=spr_error,
        spr_flag=spr_flag, spr_perc=spr_perc)

sparse_subspace_preservation_stats_jit = jit(sparse_subspace_preservation_stats)