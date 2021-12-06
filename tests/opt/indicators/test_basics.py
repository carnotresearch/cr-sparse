from indicators_setup import *


def test_zero():
    f = indicators.indicator_zero()
    assert_array_equal(f(0), 0)
    assert_array_equal(f(1), jnp.inf)
    assert_array_equal(f([0, 0]), 0)
    assert_array_equal(f([1, 0]), jnp.inf)


def test_singleton_1():
    f = indicators.indicator_singleton(1)
    assert_array_equal(f(1), 0)
    assert_array_equal(f(2), jnp.inf)
    assert_array_equal(f([1, 1]), 0)
    assert_array_equal(f([1, 0]), jnp.inf)

def test_singleton_2():
    f = indicators.indicator_singleton([1, 1])
    assert_array_equal(f([1, 1]), 0)
    assert_array_equal(f(2), jnp.inf)
    assert_array_equal(f([2, 1]), jnp.inf)
    assert_array_equal(f([1, 0]), jnp.inf)

@pytest.mark.parametrize("A", [ 
    [[1,2]],
    [[1,-2]],
    [[1,-2], [3, -4]],
])
def test_affine_1(A):
    A = jnp.asarray(A)
    m,n = A.shape
    x = jnp.arange(n)
    b = A @ x
    f = indicators.indicator_affine(A, b)
    assert_array_equal(f(x), 0)
 

def test_affine_2():
    # This A simply keeps the first component of x
    A = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    f = indicators.indicator_affine(A)
    assert_array_equal(f([0, 1, 1, -1]), 0)
    assert_array_equal(f([0, 0, 1, -10]), 0)
    assert_array_equal(f([-1, 0, 1, -10]), jnp.inf)
    assert_array_equal(f([-1, 0, 0, 0]), jnp.inf)

@pytest.mark.parametrize("b", [
    1,
    2,
    3, 
])
def test_affine_3(b):
    # This A simply keeps the first component of x
    A = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    f = indicators.indicator_affine(A, [b, 0, 0])
    assert_array_equal(f([b, 1, 1, -1]), 0)
    assert_array_equal(f([b, 0, 1, -10]), 0)
    assert_array_equal(f([b+1, 0, 1, -10]), jnp.inf)
    assert_array_equal(f([b-1, 0, 0, 0]), jnp.inf)
