from jax import jit, vmap
import jax.numpy as jnp


import cr.sparse as crs


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
    # G = jnp.minimum(G, 1)
    return jnp.rad2deg(jnp.arccos(G))


def min_angles_within_cluster(angles, cluster_sizes):