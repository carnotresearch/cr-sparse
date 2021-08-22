from .lop_setup import *


def test_gaussian():
    T = lop.gaussian_dict(keys[0], 10, 20)
    assert lop.dot_test_real(keys[1], T)

def test_rademacher():
    T = lop.rademacher_dict(keys[0], 10, 20)
    assert lop.dot_test_real(keys[1], T)

def test_random_on_rows():
    T = lop.random_orthonormal_rows_dict(keys[0], 10, 20)
    assert lop.dot_test_real(keys[1], T)

def test_random_onb():
    T = lop.random_onb_dict(keys[0], 10)
    A  = lop.to_matrix(T)
    assert crs.has_orthogonal_rows(A)
    assert crs.has_orthogonal_columns(A)
    assert lop.dot_test_real(keys[1], T)

