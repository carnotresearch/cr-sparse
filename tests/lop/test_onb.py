from .lop_setup import *


def test_fourier_basis():
    n = 8
    T = lop.jit(lop.fourier_basis(n))
    x = random.normal(keys[0], (n,))
    f = T.times(x)
    assert_allclose(crs.energy(x), crs.energy(f), atol=atol, rtol=rtol)
    a = T.trans(f)
    assert_allclose(x, a, atol=atol, rtol=rtol)
    F = lop.to_matrix(T)
    assert crs.has_unitary_rows(F)
    assert crs.has_unitary_columns(F)
    assert lop.dot_test_real(keys[0], T)

def test_cosine_basis():
    n = 8
    T = lop.jit(lop.cosine_basis(n))
    x = random.normal(keys[0], (n,))
    f = T.times(x)
    assert_allclose(crs.energy(x), crs.energy(f), atol=atol, rtol=rtol)
    a = T.trans(f)
    assert_allclose(x, a, atol=atol, rtol=rtol)
    F = lop.to_matrix(T)
    assert crs.has_unitary_rows(F)
    assert crs.has_unitary_columns(F)
    assert lop.dot_test_real(keys[0], T)

def test_wh_basis():
    n = 8
    T = lop.jit(lop.walsh_hadamard_basis(n))
    x = random.normal(keys[0], (n,))
    f = T.times(x)
    assert_allclose(crs.energy(x), crs.energy(f), atol=atol, rtol=rtol)
    a = T.trans(f)
    assert_allclose(x, a, atol=atol, rtol=rtol)
    F = lop.to_matrix(T)
    assert crs.has_unitary_rows(F)
    assert crs.has_unitary_columns(F)
    assert lop.dot_test_real(keys[0], T)

