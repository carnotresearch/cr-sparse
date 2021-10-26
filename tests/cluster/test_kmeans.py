from cluster_setup import *

# Number of points for each cluster
pts = 10
# Mean vector for first cluster
mu_a = jnp.array([0, 0])
# Covariance matrix for first cluster
cov_a = jnp.array([[4, 1], [1, 4]])
# Sampled points for the first cluster
a = random.multivariate_normal(crs.KEYS[0], mu_a, cov_a, shape=(pts,))
# Mean vector for second cluster
mu_b = jnp.array([30, 10])
# Covariance matrix for second cluster
cov_b = jnp.array([[10, 2], [2, 1]])
# Sampled points for the second cluster
b = random.multivariate_normal(crs.KEYS[1], mu_b, cov_b, shape=(pts,))
# combined points
features = jnp.concatenate((a, b))
# number of clusters
k=2
# true labels
true_labels = cluster.labels_from_sizes(jnp.ones(k, dtype=int)*pts)

def test_kmeans1():
    # Perform K-means clustering
    result = vq.kmeans_jit(crs.KEYS[3], features, k)
    pred_labels = result.assignment
    error = cluster.clustering_error_k(true_labels, pred_labels, 2)
    assert error.error == 0
