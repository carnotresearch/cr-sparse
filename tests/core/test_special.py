import pytest

import cr.sparse as crs



def test_pascal_1():
    n = 4
    A = crs.pascal_jit(n)
    assert A.shape == (n,n)
    assert A[n-1,n-1] == 1

def test_pascal_2():
    n = 4
    A = crs.pascal_jit(n, True)
    assert A.shape == (n,n)
    assert A[n-1,n-1] == 20
