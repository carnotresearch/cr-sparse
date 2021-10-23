from .lop_setup import *


def test_circulant():
    x = jnp.array([1., -1, 2, -2, 1, -1, 3, -4])
    n = len(x)
    c =  jnp.array([-1.4, 1.1, 1.2, 1.3])
    k = len(c)
    T = lop.circulant(n, c)
    expected = jnp.array([-3.5,  1.6, -7.9,  5.1, -2.5,  2.7, -6.7,  9. ])
    assert_allclose(T.times(x), expected, atol=atol, rtol=rtol)
    A = lop.to_matrix(T)
    AT = lop.to_adjoint_matrix(T)
    assert_allclose(A.T, AT, atol=atol, rtol=rtol)
    for i in range(n -k):
        assert_allclose(A[i:i+k, i], c, atol=atol, rtol=rtol)
    #assert lop.dot_test_real(keys[0], T, tol=1e-5)

