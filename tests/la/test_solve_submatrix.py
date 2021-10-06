from .setup import *

@pytest.mark.parametrize("K", [1, 2, 4, 8])
def test_solve1(K):
    M = 20
    N = 40
    Phi = crdict.gaussian_mtx(keys[0], M, N)
    cols = random.permutation(keys[1], jnp.arange(N))[:K]
    X = random.normal(keys[2], (K, 1))
    Phi_I = Phi[:, cols]
    B_ref = Phi_I @ X
    B = crla.mult_with_submatrix(Phi, cols, X)
    assert_allclose(B_ref, B)
    Z, R = crla.solve_on_submatrix(Phi, cols, B)
    assert_allclose(Z, X, atol=atol, rtol=rtol)


submat_multiplier = vmap(crla.mult_with_submatrix, (None, 1, 1), 1)
submat_solver = vmap(crla.solve_on_submatrix, (None, 1, 1), (1, 1,))

@pytest.mark.parametrize("K", [1, 2, 4])
def test_solve2(K):
    M = 20
    N = 40
    Phi = crdict.gaussian_mtx(keys[0], M, N)
    # Number of signals
    S = 4
    # index sets for each signal
    cols = crdata.index_sets(keys[1], N, K, S, 1)
    # signals [column wise]
    X = random.normal(keys[2], (K, S))
    # measurements
    B = submat_multiplier(Phi, cols, X)
    # solutions
    Z, R = submat_solver(Phi, cols, B)
    # verify
    assert_allclose(Z, X, atol=atol, rtol=rtol)
