
from projectors_setup import *

def test_zero():
    proj = projectors.proj_zero()
    assert_array_equal(proj(0), 0)
    assert_array_equal(proj(1), 0)
    assert_array_equal(proj([1,1]), [0, 0])


def test_singleton():
    proj = projectors.proj_singleton(1)
    assert_array_equal(proj(0), 1)
    assert_array_equal(proj(1), 1)
    assert_array_equal(proj(2), 1)
    assert_array_equal(proj([1,1]), [1, 1])
    assert_array_equal(proj([1,2, 3, 4]), [1, 1, 1, 1])


@pytest.mark.parametrize("x", [
    0,
    1,
    [2,3],
    [[1,2], [3,4]]
])
def test_identity(x):
    proj = projectors.proj_identity()
    assert_array_equal(proj(x), x)


@pytest.mark.parametrize("x", [
    [1],
    [2,3],
    [1,2, 3, 4]
])
def test_affine_1(x):
    n = len(x)
    A = jnp.eye(n)
    b = 0
    proj = projectors.proj_affine(A, b)
    v = proj(x)
    assert_allclose(A @ v - b, 0)

@pytest.mark.parametrize("x", [
    [1],
    [2,3],
    [1,2, 3, 4]
])
def test_affine_2(x):
    n = len(x)
    A = random.normal(crs.KEYS[0], (n,n))
    b = 0
    proj = projectors.proj_affine(A, b)
    v = proj(x)
    assert_allclose(A @ v - b, 0, atol=atol, rtol=rtol)
