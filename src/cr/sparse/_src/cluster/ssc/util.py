from jax import jit, vmap
import jax.numpy as jnp


import cr.sparse as crs
import cr.sparse.cluster as crcluster


@jit
def sparse_to_full_rep(X, I):
    """Combines values and indices arrays to sparse representations
    """
    # number of signals
    n  = X.shape[1]
    mapper = lambda x, i : jnp.zeros(n).at[i].set(x)
    return vmap(mapper, (1,1), 1)(X, I)


def angles_between_points(X):
    """Returns an SxS matrix of angles between each pair of points
    """
    # make sure that the points are normalized
    X = crs.normalize_l2_cw(X)
    # Compute gram matrix
    G = X.T @ X
    # Avoid overflow in gram matrix
    G = jnp.minimum(G, 1)
    return jnp.rad2deg(jnp.arccos(G))


def min_angles_inside_cluster(angles, cluster_sizes):
    """Returns the minimum angles for for each point with its neighbors inside the cluster 
    """
    # we have to ignore the diagonal elements
    angles = crs.set_diagonal(angles, 10000)
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
    angles = crs.set_diagonal(angles, 10000)
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