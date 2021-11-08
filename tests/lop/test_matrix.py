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



def test_matrix_3():
    m, n, k = 10, 20, 5
    # axis=0 [process column wise]
    A = random.normal(crs.KEYS[0], (m,n))
    T = lop.jit(lop.matrix(A))
    X = random.normal(crs.KEYS[1], (n,k))
    assert_allclose(T.times(X), A @ X)
    Y = random.normal(crs.KEYS[2], (m,k))
    assert_allclose(T.trans(Y), A.T @ Y)

    # axis=1 [process row wise]
    T = lop.jit(lop.matrix(A, axis=1))
    X = X.T
    Y = Y.T
    assert_allclose(T.times(X), (X @ A.T), atol=atol)
    assert_allclose(T.trans(Y), (Y @ A), atol=atol)


def test_matrix_4():
    # complex case
    m, n, k = 10, 20, 5
    # axis=0 [process column wise]
    Ar = random.normal(crs.KEYS[0], (m,n))
    Ac = random.normal(crs.KEYS[1], (m,n))
    A = Ar + Ac * 1j
    T = lop.jit(lop.matrix(A))

    Xr = random.normal(crs.KEYS[2], (n,k))
    Xc = random.normal(crs.KEYS[3], (n,k))
    X = Xr + Xc * 1j
    assert_allclose(T.times(X), A @ X)
    Yr = random.normal(crs.KEYS[4], (m,k))
    Yc = random.normal(crs.KEYS[5], (m,k))
    Y = Yr + Yc * 1j
    assert_allclose(T.trans(Y), crs.hermitian(A) @ Y)

    # axis=1 [process row wise]
    T = lop.jit(lop.matrix(A, axis=1))
    X = X.T
    Y = Y.T
    assert_allclose(T.times(X), (X @ A.T), atol=atol)
    assert_allclose(T.trans(Y), (Y @ jnp.conjugate(A)), atol=atol)

