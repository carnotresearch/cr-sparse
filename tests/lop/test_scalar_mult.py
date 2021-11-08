from .lop_setup import *


@pytest.mark.parametrize("alpha,n", [
    [1, 10],
    [3., 10],
    [3. + 4j, 10],
])
def test_scalar_mult(alpha,n):
    T = lop.scalar_mult(alpha, n)
    T = lop.jit(T)
    x = random.normal(crs.KEYS[1], (n,))
    y = T.times(x)
    assert_array_equal(y, alpha * x)
    assert lop.dot_test_real(keys[0], T)
    assert lop.dot_test_complex(keys[0], T)
