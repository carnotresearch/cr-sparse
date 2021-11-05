from .lop_setup import *


@pytest.mark.parametrize("A", [ 
    jnp.reshape(jnp.arange(20), (4,5)),
    [[1., 3j], [2j, 5]],
    [[1., 3j], [2j, 5]],
    [[1., 3j], [2j, 5]],
])
def test_scale(A):
    A = jnp.asarray(A)
    m, n = A.shape
    T = lop.jit(lop.matrix(A))
    TH = lop.jit(lop.adjoint(T))

    assert_allclose(lop.to_adjoint_matrix(T), lop.to_matrix(TH))
    assert_allclose(lop.to_adjoint_matrix(TH), lop.to_matrix(T))
    assert lop.dot_test_real(keys[2], T)
    assert lop.dot_test_real(keys[3], TH)
    assert lop.dot_test_complex(keys[2], T)
    assert lop.dot_test_complex(keys[3], TH)
