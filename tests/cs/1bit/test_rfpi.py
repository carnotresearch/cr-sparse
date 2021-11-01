from cs1bit_setup import *


def test_rfpi():
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
    # initial guess
    x0 = cs1bit.rfp_random_guess(Phi)
    x0 = cs1bit.rfp_lsqr_guess(Phi, y)
    # solution
    delta = 0.1
    lambda_ = M
    inner_iters=2
    outer_iters=2
    state = cs1bit.rfp(Phi, y, x0, lambda_=lambda_, delta=delta, 
        inner_iters=inner_iters, outer_iters=outer_iters)


