from ssc_setup import *

# ambient space dimension
N = 9
# subspace dimension
D = 3
# number of subspaces
K = 5
# number of points per subspace   
S = 20

bases = crdata.random_subspaces_jit(crs.KEYS[0], N, D, K)
X = crdata.uniform_points_on_subspaces_jit(crs.KEYS[1], bases, S)
sizes = jnp.repeat(S, K)
true_labels = jnp.repeat(jnp.arange(K), S)


def test_build_omp_rep():
    Z, I, R = ssc.build_representation_omp_jit(X, D)
    r_norms = crs.norms_l2_cw(R)
    #print(r_norms)
    assert_allclose(r_norms, 0, atol=1e-1)
    Z_full = ssc.sparse_to_full_rep(Z, I)
    Z_sp = ssc.sparse_to_bcoo(Z, I)
    assert_allclose(Z_full, Z_sp.todense())
    ZZ, II = ssc.bcoo_to_sparse_jit(Z_sp, D)
    assert_allclose(Z, ZZ)
    assert_allclose(I, II)
    affinity_full = ssc.rep_to_affinity(Z_full)
    affinity_sp = ssc.rep_to_affinity(Z_sp)
    assert_allclose(affinity_full, affinity_sp.todense())
    spr_stats = ssc.subspace_preservation_stats_jit(Z_full, true_labels)
    spr_stats = ssc.sparse_subspace_preservation_stats_jit(Z, I, true_labels)
    stats = str(spr_stats)


def test_batch_build_omp_rep():
    Z, I, R = ssc.batch_build_representation_omp_jit(X, D, 10)
    r_norms = crs.norms_l2_cw(R)
    assert_allclose(r_norms, 0, atol=1e-1)

def test_angles_between_points():
    angles = ssc.angles_between_points(X)
    d = jnp.diag(angles)
    # for 32-bit computation the floating point error is high
    assert_allclose(d, 0., atol=1e-1)
    min_inside = ssc.min_angles_inside_cluster(angles, sizes)
    assert_array_equal(min_inside > 0, True)
    min_outside = ssc.min_angles_outside_cluster(angles, sizes)
    assert_array_equal(min_outside > 0, True)
    inn_indices = ssc.nearest_neighbors_inside_cluster(angles, sizes)
    onn_indices = ssc.nearest_neighbors_outside_cluster(angles, sizes)
    neighbors = ssc.sorted_neighbors(angles)
    positions = ssc.inn_positions(true_labels, neighbors)
