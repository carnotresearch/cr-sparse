from .lop_setup import *


def test_hcat():
    A = jnp.ones((4,5))
    A = lop.matrix(A)
    B = lop.identity(4)
    C = lop.hcat(A, B)
    x1 = jnp.arange(5)
    x2 = jnp.arange(6,10)
    x = jnp.concatenate((x1, x2))
    a = C.times(x)
    b = A.times(x1) + B.times(x2)
    assert_allclose(a, b)
    assert lop.dot_test_real(keys[0], A)
    assert lop.dot_test_real(keys[1], B)
    assert lop.dot_test_real(keys[2], C)

def test_negation():
    A = jnp.reshape(jnp.arange(20), (4,5))
    A = lop.matrix(A)
    B = -A
    x = jnp.arange(5)
    assert_allclose(A.times(x), -B.times(x))
    assert lop.dot_test_real(keys[0], A)
    assert lop.dot_test_real(keys[1], B)


def test_scale():
    A = jnp.reshape(jnp.arange(20), (4,5))
    A = lop.jit(lop.matrix(A))
    alpha = 4.
    B = lop.scale(A, alpha)
    B = lop.jit(B)
    x = jnp.arange(5)
    assert_allclose(alpha*A.times(x), B.times(x))
    assert lop.dot_test_real(keys[0], A)
    assert lop.dot_test_real(keys[1], B)


def test_composition():
    m, n, p = 3, 4, 5
    A = lop.matrix(random.normal(keys[0], (m,n)))
    B = lop.matrix(random.normal(keys[1], (n,p)))
    C = lop.jit(A @ B)
    assert lop.dot_test_real(keys[2], A)
    assert lop.dot_test_real(keys[3], B)
    assert lop.dot_test_real(keys[4], C)

    x = random.normal(keys[5], (p,))
    assert_allclose(C.times(x), A.times(B.times(x)))


def test_exponentiation():
    n = 4
    A = lop.matrix(random.normal(keys[0], (n,n)))
    B = A ** 3
    assert lop.dot_test_real(keys[2], A)
    assert lop.dot_test_real(keys[3], B)

    x = random.normal(keys[5], (n,))
    assert_allclose(B.times(x), A.times(A.times(A.times(x))))
