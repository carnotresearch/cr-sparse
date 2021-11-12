from .lop_setup import *




@pytest.mark.parametrize("A,alpha", [ 
    (jnp.reshape(jnp.arange(20), (4,5)), 4.),
    ([[1., 3j], [2j, 5]], 4.),
    ([[1., 3j], [2j, 5]], 4. + 2.j),
    ([[1., 3j], [2j, 5]], 2 + 3j),
])
def test_scale(A, alpha):
    A = jnp.asarray(A)
    m, n = A.shape
    TA = lop.jit(lop.matrix(A))
    TB1 = lop.scale(TA, alpha)
    TB1 = lop.jit(TB1)
    B = alpha * A 
    TB2 = lop.matrix(B)
    TB2 = lop.jit(TB2)
    x = random.normal(keys[2], (n,))
    assert TA.real == jnp.isrealobj(A)
    assert_allclose(alpha*TA.times(x), TB1.times(x))
    assert_allclose(TB1.times(x), TB2.times(x))

    y = random.normal(keys[2], (m,))
    assert_allclose(TB1.trans(y), TB2.trans(y), atol=atol)

    assert lop.dot_test_real(keys[0], TA)
    assert lop.dot_test_real(keys[1], TB1)
    assert lop.dot_test_real(keys[1], TB2)
    assert lop.dot_test_complex(keys[2], TB1)
    assert lop.dot_test_complex(keys[2], TB2)
