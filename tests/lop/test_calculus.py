from .lop_setup import *

def test_first_derivative_forward():
    x = jnp.array([1, 3, 4, -5, -7, 8, -10, 15])
    n = len(x)
    T = lop.jit(lop.first_derivative(n, kind='forward'))
    A = lop.to_matrix(T)
    AT = lop.to_adjoint_matrix(T)
    assert jnp.allclose(A.T, AT)
    e1 = jnp.array([  2.,   1.,  -9.,  -2.,  15., -18.,  25.,   0.])
    assert_allclose(T.times(x), e1)
    assert_allclose(A @ x, e1)
    e2 = jnp.array([ -1.,  -2.,  -1.,   9.,   2., -15.,  18., -10.])
    assert_allclose(T.trans(x), e2)
    assert_allclose(AT @ x, e2)
    assert lop.dot_test_real(keys[0], T)
    d = jnp.array([-1, 1])
    for i in range(n-1):
        assert_allclose(A[i, i:i+2],d)
    for i in range(n-1):
        assert_allclose(AT[i:i+2, i],d)

def test_first_derivative_backward():
    x = jnp.array([1, 3, 4, -5, -7, 8, -10, 15])
    n = len(x)
    T = lop.jit(lop.first_derivative(n, kind='backward'))
    A = lop.to_matrix(T)
    AT = lop.to_adjoint_matrix(T)
    assert jnp.allclose(A.T, AT)
    e1 = jnp.array([  0.,   2.,   1.,  -9.,  -2.,  15., -18.,  25.])
    assert_allclose(T.times(x), e1)
    assert_allclose(A @ x, e1)
    e2 = jnp.array([ -3.,  -1.,   9.,   2., -15.,  18., -25.,  15.])
    assert_allclose(T.trans(x), e2)
    assert_allclose(AT @ x, e2)
    assert lop.dot_test_real(keys[0], T)
    d = jnp.array([-1, 1])
    for i in range(1, n):
        assert_allclose(A[i, i-1:i+1],d)
    for i in range(1, n):
        assert_allclose(AT[i-1:i+1, i],d)

def test_first_derivative_centered():
    x = jnp.array([1, 3, 4, -5, -7, 8, -10, 15])
    n = len(x)
    T = lop.jit(lop.first_derivative(n, kind='centered'))
    A = lop.to_matrix(T)
    AT = lop.to_adjoint_matrix(T)
    assert jnp.allclose(A.T, AT)
    e1 = jnp.array([  0. ,  1.5, -4. , -5.5,  6.5, -1.5,  3.5,  0.])
    assert_allclose(T.times(x), e1)
    assert_allclose(A @ x, e1)
    e2 = jnp.array([ -1.5, -2. ,  4. ,  5.5, -6.5,  1.5,  4. , -5.])
    assert_allclose(T.trans(x), e2)
    assert_allclose(AT @ x, e2)
    assert lop.dot_test_real(keys[0], T)
    d = jnp.array([-1, 1]) / 2.
    for i in range(1, n-1):
        assert_allclose(A[i, i-1:i+2:2],d)
    for i in range(1, n-1):
        assert_allclose(AT[i-1:i+2:2, i],d)


def test_second_derivative():
    x = jnp.array([1, 3, 4, -5, -7, 8, -10, 15])
    n = len(x)
    T = lop.jit(lop.second_derivative(n))
    A = lop.to_matrix(T)
    AT = lop.to_adjoint_matrix(T)
    assert jnp.allclose(A.T, AT)
    d = jnp.array([1, -2, 1])
    for i in range(1, n-1):
        assert_allclose(A[i, i-1:i+2],d)
    for i in range(1, n-1):
        assert_allclose(AT[i-1:i+2, i],d)
    e1 = jnp.array([0.,  -1., -10.,   7.,  17., -33.,  43.,   0])
    e2 = jnp.array([3.,  -2., -10.,   7.,  17., -33.,  28., -10.])
    assert_allclose(T.times(x), e1)
    assert_allclose(A @ x, e1)
    assert_allclose(T.trans(x), e2)
    assert_allclose(AT @ x, e2)
