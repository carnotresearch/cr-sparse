from .lop_setup import *



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
    Z = lop.zero(n, m)
    x = random.normal(keys[0], (m,))
    y = random.normal(keys[0], (n,))
    assert_allclose(Z.times(y), jnp.zeros((m,)))
    assert_allclose(Z.trans(x), jnp.zeros((n,)))
    A = lop.to_matrix(Z)
    A2 = lop.to_complex_matrix(Z)
    assert_allclose(A, A2)

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
    assert lop.dot_test_real(keys[0], F)
    assert lop.dot_test_complex(keys[0], F)


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
    assert lop.dot_test_real(keys[0], T)
    assert lop.dot_test_complex(keys[0], T)


def test_zero_padding():
    n = 4
    T = lop.jit(lop.pad_zeros(n, 2, 2))
    assert not T.matrix_safe
    x = jnp.arange(n) + 1.
    y = T.times(x)
    z = jnp.zeros(2)
    yy = jnp.concatenate((z, x, z))
    assert_allclose(y, yy) 
    k = 8
    X = jnp.reshape(jnp.arange(n*k), (n,k)) + 1.
    Z = jnp.zeros((2, k))
    Y  = jnp.vstack((Z, X, Z))
    assert_allclose(T.times_2d(X), Y) 
    assert lop.dot_test_real(keys[0], T)
 

def test_real_part():
    n = 4
    T = lop.jit(lop.real(n))
    x = jnp.arange(n) + 1.
    xx = x + 4j
    assert_allclose(T.times(xx), x)
    assert_allclose(T.trans(T.times(xx)), x)
    assert lop.dot_test_real(keys[0], T)
    assert lop.dot_test_complex(keys[0], T)


def test_symmetrize():
    n = 4
    T = lop.jit(lop.symmetrize(n))
    x = random.normal(keys[0], (n,))
    y = jnp.concatenate((x[::-1], x))
    assert_allclose(T.times(x), y)
    assert_allclose(T.trans(T.times(x)), 2*x)
    assert lop.dot_test_real(keys[1], T)
    assert lop.dot_test_complex(keys[1], T)


