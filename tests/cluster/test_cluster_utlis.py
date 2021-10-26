from cluster_setup import *


def test_sizes_from_labels():
    labels = jnp.array([0, 0, 1, 1, 2, 2])
    sizes = cluster.sizes_from_labels_jit(labels, 3)
    assert_array_equal(sizes, jnp.array([2, 2, 2]))


def test_start_end_indices():
    sizes = jnp.array([4, 4])
    starts, ends = cluster.start_end_indices(sizes)
    assert_array_equal(starts, jnp.array([0, 4]))
    assert_array_equal(ends, jnp.array([4, 8]))


def test_labels_from_sizes():
    sizes = jnp.array([2, 2])
    labels = cluster.labels_from_sizes(sizes)
    assert_array_equal(labels, jnp.array([0, 0, 1, 1]))

def test_best_map():
    true_labels = jnp.array([0, 0, 1, 1])
    pred_labels = jnp.array([1, 1, 0, 0])
    mapped_labels, cols, G  = cluster.best_map(true_labels, pred_labels)
    assert_array_equal(mapped_labels, true_labels)


def test_best_map_k():
    true_labels = jnp.array([0, 0, 1, 1])
    pred_labels = jnp.array([1, 1, 0, 0])
    mapped_labels, cols, G  = cluster.best_map_k(true_labels, pred_labels, 2)
    assert_array_equal(mapped_labels, true_labels)


def test_cluster_error():
    true_labels = jnp.array([0, 0, 1, 1])
    pred_labels = jnp.array([1, 1, 0, 0])
    error = cluster.clustering_error(true_labels, pred_labels)
    assert error.error == 0
    assert error.error_perc == 0
    s = str(error)

def test_cluster_error_k():
    true_labels = jnp.array([0, 0, 1, 1])
    pred_labels = jnp.array([1, 1, 0, 0])
    error = cluster.clustering_error_k(true_labels, pred_labels, 2)
    assert error.error == 0
    assert error.error_perc == 0
