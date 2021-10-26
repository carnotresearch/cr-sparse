from cluster_setup import *
from jax.experimental.sparse import BCOO
num_clusters = 3
gap = 3
points_per_set = 50
total = points_per_set*num_clusters
means = jnp.arange(num_clusters) * gap
means = jnp.repeat(means, points_per_set)
points = random.uniform(crs.KEYS[0], (total,))
points = points  - 0.5
points = points + means
true_labels = jnp.repeat(jnp.arange(num_clusters), points_per_set)
sqr_distances = crs.pdist_sqr_l2_rw(points[:, jnp.newaxis])
sigma = .5
similarity = crs.sqr_dist_to_gaussian_sim(sqr_distances, sigma)

def test_spectral1():
    res = spectral.unnormalized(crs.KEYS[1], similarity)
    pred_labels =  res.assignment
    error = cluster.clustering_error(true_labels, pred_labels)
    assert error.error == 0


def test_spectral2():
    res = spectral.unnormalized_k_jit(crs.KEYS[1], similarity, num_clusters)
    pred_labels =  res.assignment
    error = cluster.clustering_error(true_labels, pred_labels)
    assert error.error == 0

def test_spectral3():
    res = spectral.normalized_random_walk(crs.KEYS[1], similarity)
    pred_labels =  res.assignment
    error = cluster.clustering_error(true_labels, pred_labels)
    assert error.error == 0


def test_spectral4():
    res = spectral.normalized_random_walk_k_jit(crs.KEYS[1], similarity, num_clusters)
    pred_labels =  res.assignment
    error = cluster.clustering_error(true_labels, pred_labels)
    assert error.error == 0


def test_spectral5():
    res = spectral.normalized_symmetric_fast_k_jit(crs.KEYS[1], similarity, num_clusters)
    pred_labels =  res.assignment
    error = cluster.clustering_error(true_labels, pred_labels)
    assert error.error == 0


def test_spectral6():
    m = 10
    a = jnp.ones((m, m))
    z = jnp.zeros((m, m))
    az = jnp.hstack((a, z))
    za = jnp.hstack((z, a))
    affinity = jnp.vstack((az, za))
    print(affinity)
    true_labels = cluster.labels_from_sizes(jnp.array([m, m]))
    affinity = BCOO.fromdense(affinity)
    k = 2
    res = spectral.normalized_symmetric_sparse_fast_k_jit(crs.KEYS[1], affinity, k)
    pred_labels =  res.assignment
    error = cluster.clustering_error(true_labels, pred_labels)
    assert error.error == 0
