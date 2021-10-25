from sls_setup import *


def test_lanbpro1():
    A =  jnp.eye(4)
    T = lop.matrix(A)
    r = lasvd.lanbpro_random_start(crs.KEYS[0], T)
    state = crsvd.lanbpro_jit(T, 4, r)
    s = str(state)
    assert_allclose(state.alpha, 1., atol=atol)
    assert_allclose(state.beta[1:], 0., atol=atol)


def test_lansvd1():
    A =  jnp.eye(4)
    T = lop.matrix(A)
    r = lasvd.lanbpro_random_start(crs.KEYS[0], A)
    U, S, V, bnd, n_converged, state = crsvd.lansvd_simple_jit(T, 4, r)
    assert_allclose(state.alpha, 1., atol=atol)
    assert_allclose(state.beta[1:], 0., atol=atol)
