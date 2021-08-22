from .lop_setup import *

def test_dirac_fourier_basis():
    n = 8
    F = lop.jit(lop.fourier_basis(n))
    T = lop.jit(lop.dirac_fourier_basis(n))
    assert T.shape == (n, 2*n)
    x1 = random.normal(keys[0], (n,))
    x2 = random.normal(keys[1], (n,))
    x = jnp.concatenate((x1, x2))
    f = T.times(x)
    assert_allclose(f, F.times(x2) + x1)
    assert lop.dot_test_real(keys[0], T)
