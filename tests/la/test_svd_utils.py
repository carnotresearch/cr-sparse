from .setup import *

# ambient dimension
n = 10
# subspace dimension
d = 4
# number of subspaces
k = 2

bases = crdata.random_subspaces_jit(crs.KEYS[0], n, d, k)


def test_orth():
    A = jnp.array([[2, 0, 0], [0, 5, 0]])  # rank 2 array
    Q, rank = crla.orth_jit(A)
    Q0 = jnp.array([[0., 1.], [1., 0.]])
    assert_allclose(Q, Q0)

def test_row_space():
    A = jnp.array([[2, 0, 0], [0, 5, 0]])  # rank 2 array
    Q, rank = crla.row_space_jit(A)
    Q0 = jnp.array([[0., 1.], [1., 0.], [0, 0]])
    assert_allclose(Q, Q0)

def test_null_space():
    A = bases[0]
    Z, r = crla.null_space_jit(A)
    Z = Z[:, r:]
    assert_allclose(A @ Z, 0)

def test_left_null_space():
    A = bases[0]
    Z, r = crla.left_null_space_jit(A)
    Z = Z[:, r:]
    assert_allclose(Z.T @ A, 0, atol=atol)

def test_effective_rank():
    A = random.normal(key, (3, 5))
    r = crla.effective_rank_jit(A)
    assert_array_equal(r, 3)

def test_singular_values():
    A = bases[0]
    s = crla.singular_values(A)
    assert_allclose(s, 1., atol=atol)