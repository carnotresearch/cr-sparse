from .lop_setup import *


def test_matrix_1():
    A = jnp.ones([4, 4])
    op = lop.matrix(A)
    x = jnp.ones(4)
    assert_allclose(op.times(x), A @ x)
    assert_allclose(op.trans(x), A.T @ x)

def test_matrix_2():
    A = jnp.reshape(jnp.arange(20), (4,5))
    op = lop.matrix(A)
    x = jnp.ones(4)
    y = jnp.ones(5)
    assert_allclose(op.times(y), A @ y)
    assert_allclose(op.trans(x), A.T @ x)
    assert lop.dot_test_real(keys[0], op)


def test_diag():
    n = 4
    x = jnp.arange(n)+1.
    T = lop.diagonal(x)
    X = lop.to_matrix(T)
    assert_allclose(X, jnp.diag(x))
    y = T.times(x)
    assert_allclose(y, x**2)
    assert lop.dot_test_real(keys[0], T)


def test_zero():
    m, n = 4, 5
    Z = lop.zero(m, n)
    x = random.normal(keys[0], (m,))
    y = random.normal(keys[0], (n,))
    assert_allclose(Z.times(y), jnp.zeros((m,)))
    assert_allclose(Z.trans(x), jnp.zeros((n,)))


def test_flipud():
    n = 4
    x = jnp.arange(n) + 1.
    k = 8
    X = jnp.reshape(jnp.arange(n*k), (n,k)) + 1.
    F = lop.flipud(n)
    assert_allclose(F.times(x), x[::-1])
    assert_allclose(F.times(X), X[::-1, :])
    assert_allclose(F.trans(x), x[::-1])
    assert_allclose(F.trans(X), X[::-1, :])
    assert_allclose(F.times(F.times(x)), x)
    assert_allclose(F.trans(F.times(x)), x)
    assert_allclose(F.times(F.times(X)), X)
    assert_allclose(F.trans(F.times(X)), X)
    lop.dot_test_real(keys[0], F)
    lop.dot_test_complex(keys[0], F)


def test_sum():
    n = 4
    T = lop.jit(lop.sum(n))
    A = lop.to_matrix(T)
    assert_allclose(A, jnp.ones((1,4)))
    A = lop.to_adjoint_matrix(T)
    assert_allclose(A, jnp.ones((4,1)))
    x = jnp.arange(n) + 1.
    s = jnp.sum(x)
    assert_allclose(T.times(x), s) 
    assert_allclose(T.trans(s), jnp.ones(n) * s) 
    lop.dot_test_real(keys[0], T)
    lop.dot_test_complex(keys[0], T)
