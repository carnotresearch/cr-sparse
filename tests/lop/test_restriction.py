from .lop_setup import *


def test_resriction_0():
    n = 4
    index = jnp.array([0, 2])
    T = lop.jit(lop.restriction(n, index))
    x = random.normal(keys[0], (n,))
    assert_allclose(T.times(x), x[index])
    y = jnp.zeros_like(x).at[index].set(x[index])
    assert lop.dot_test_real(keys[0], T)
    assert_allclose(T.trans(T.times(x)), y)
    assert lop.dot_test_complex(keys[1], T)


@pytest.mark.parametrize("n,k,s,i", [
    (10, 4, 1, 0),
    (10, 4, 5, 1),
    (20, 10, 5, 4),
])
def test_restriction1(n, k, s, i):
    I = random.permutation(crs.KEYS[i], n)
    I = I[:k]
    # column-wise work
    T = lop.restriction(n, I, axis=0)
    x = random.normal(crs.KEYS[i], (n,s))
    y = T.times(x)
    y_ref = x[I, :]
    assert_array_equal(y_ref, y)
    y2 = T.trans(y)
    y3 = T.times(y2)
    assert_array_equal(y_ref, y3)
    assert lop.dot_test_real(crs.KEYS[i+1], T, tol=1e-4)
    assert lop.dot_test_complex(crs.KEYS[i+1], T, tol=1e-4)
    # row-wise work
    T = lop.restriction(n, I, axis=1)
    x = random.normal(crs.KEYS[i], (s,n))
    y = T.times(x)
    y_ref = x[:, I]
    assert_array_equal(y_ref, y)
    y2 = T.trans(y)
    y3 = T.times(y2)
    assert_array_equal(y_ref, y3)
    assert lop.dot_test_real(crs.KEYS[i+2], T, tol=1e-4)
    assert lop.dot_test_complex(crs.KEYS[i+2], T, tol=1e-4)
