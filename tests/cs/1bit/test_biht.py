from cs1bit_setup import *


def test_biht():
    M, N = 256, 512
    K = 4
    # dictionary
    Phi = crdict.gaussian_mtx(crs.KEYS[0], M, N, normalize_atoms=False)
    # sparse signal
    x, omega = crdata.sparse_normal_representations(crs.KEYS[1], N, K)
    # normalize signal
    x = x / norm(x)
    # frame bound
    s0 = crdict.upper_frame_bound(Phi)
    # measurements
    y = cs1bit.measure_1bit(Phi, x)
    # solver step-size
    tau = 0.98 * s0
    # solution
    state = cs1bit.biht_jit(Phi, y, K, tau)
    # recovered support
    I = jnp.sort(state.I)
    # check if the support is recovered correctly
    assert_array_equal(omega, I)
    # reconstructed signal
    x_rec = crs.build_signal_from_indices_and_values(N, state.I, state.x_I)
    # normalize recovered signal
    x_rec = x_rec / norm(x_rec)
    # assert that recovered signal is close to original
    assert_almost_equal(norm(x - x_rec), 0, decimal=1)


