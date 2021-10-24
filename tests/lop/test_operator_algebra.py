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
    assert lop.dot_test_real(keys[0], A, tol=1e-4)
    assert lop.dot_test_real(keys[1], B, tol=1e-4)


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


def test_partial_op_picks():
    m, n = 4,5
    A = jnp.reshape(jnp.arange(20), (m, n))
    T = lop.jit(lop.matrix(A))
    picks = jnp.array([1,2])
    TP = lop.partial_op(T, picks)
    FP = lop.to_matrix(TP)
    assert_allclose(FP, A[picks])
    FPT = lop.to_adjoint_matrix(TP)
    assert_allclose(FPT, A[picks].T)
    assert_allclose(FP.T, FPT)
    assert lop.dot_test_real(keys[2], TP)

def test_partial_op_picks_perms():
    m, n = 4,5
    A = jnp.reshape(jnp.arange(20), (m, n))
    T = lop.jit(lop.matrix(A))

    picks = jnp.array([1,2])
    perm = random.permutation(key, n)
    TP = lop.partial_op(T, picks, perm)
    FP = lop.to_matrix(TP)
    FPT = lop.to_adjoint_matrix(TP)
    assert_allclose(FP.T, FPT)

    FPM = jnp.empty(A.shape, A.dtype)
    FPM = FPM.at[:, perm].set(A)
    assert_allclose(FP, FPM[picks])
    assert_allclose(FPT, FPM[picks].T)
    assert lop.dot_test_real(keys[2], TP)

def test_add():
    m, n = 4,6
    A = random.normal(keys[0], (m,n))
    B = random.normal(keys[0], (m,n))
    TA = lop.matrix(A)
    TB = lop.matrix(B)
    TC = lop.add(TA, TB)
    FC = lop.to_matrix(TC)
    FCT  = lop.to_adjoint_matrix(TC)
    assert_allclose(FC, A+B)
    assert_allclose(FC.T, FCT)
    assert lop.dot_test_real(keys[2], TC)
    assert(lop.to_matrix(TA+TB), FC)

def test_subtract():
    m, n = 4,6
    A = random.normal(keys[0], (m,n))
    B = random.normal(keys[0], (m,n))
    TA = lop.matrix(A)
    TB = lop.matrix(B)
    TC = lop.subtract(TA, TB)
    FC = lop.to_matrix(TC)
    FCT  = lop.to_adjoint_matrix(TC)
    assert_allclose(FC, A-B)
    assert_allclose(FC.T, FCT)
    assert lop.dot_test_real(keys[2], TC)
    assert(lop.to_matrix(TA-TB), FC)

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


def test_transpose():
    m, n = 4,6
    A = random.normal(keys[0], (m,n))
    T = lop.matrix(A)
    TT = lop.transpose(T)
    F = lop.to_matrix(TT)
    FT  = lop.to_adjoint_matrix(TT)
    assert_allclose(F, A.T)
    assert_allclose(FT, A)
    assert_allclose(FT, F.T)

def test_hermitian():
    m, n = 4,6
    A = random.normal(keys[0], (m,n))
    T = lop.matrix(A)
    TT = lop.hermitian(T)
    F = lop.to_matrix(TT)
    FT  = lop.to_adjoint_matrix(TT)
    assert_allclose(F, A.T)
    assert_allclose(FT, A)
    assert_allclose(FT, F.T)

def test_exponentiation():
    n = 4
    A = lop.matrix(random.normal(keys[0], (n,n)))
    B = A ** 3
    assert lop.dot_test_real(keys[2], A)
    assert lop.dot_test_real(keys[3], B)

    x = random.normal(keys[5], (n,))
    assert_allclose(B.times(x), A.times(A.times(A.times(x))))


def test_column():
    m, n = 4, 8
    A = random.normal(keys[0], (m,n))
    T = lop.matrix(A)
    assert_allclose(lop.column(T, 1), A[:,1])
    indices = (1,2)
    assert_allclose(lop.columns(T, indices), A[:,indices])
