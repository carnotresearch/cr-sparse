from .lop_setup import *


@pytest.mark.parametrize("n", [
    2,
    4,
    6,
    8,
    9,
    16,
    20,
    32,
    49
])
def test_fft(n):
    T = lop.jit(lop.fft(n))
    A = lop.to_matrix(T)
    AH = lop.to_adjoint_matrix(T)
    assert_allclose(jnp.conjugate(A).T, AH, atol=atol)
    T = lop.jit(lop.fft(n, mode='c2c'))
    A = lop.to_matrix(T)
    AH = lop.to_adjoint_matrix(T)
    assert_allclose(jnp.conjugate(A).T, AH, atol=atol)
 
