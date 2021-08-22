from .lop_setup import *


def test_running_average():
    n = 10
    k = 5
    h = jnp.ones(k) / k
    T = lop.jit(lop.running_average(n, k))
    A = lop.to_matrix(T)
    AT = lop.to_adjoint_matrix(T) 
    assert jnp.allclose(A.T, AT)
    assert crs.is_symmetric(A)

    x = random.normal(keys[0], (n,))
    y = jnp.convolve(x, h, 'same')
    assert(y, T.times(x))
    assert(y, T.trans(x))

    assert lop.dot_test_real(keys[0], T)
 

def test_fir_filter():
    h  = jnp.array([1., 2., 1.5, -1.5, 3.])
    n = 10
    T = lop.jit(lop.fir_filter(n, h))
    A = lop.to_matrix(T)
    AT = lop.to_adjoint_matrix(T) 
    assert jnp.allclose(A.T, AT)
 
    x = random.normal(keys[0], (n,))
    y = jnp.convolve(x, h, 'same')
    assert(y, T.times(x))
    assert lop.dot_test_real(keys[0], T)
