
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
    A = random.normal(cnb.KEYS[0], (n,n))
    b = 0
    proj = projectors.proj_affine(A, b)
    v = proj(x)
    assert_allclose(A @ v - b, 0, atol=atol, rtol=rtol)


@pytest.mark.parametrize("l", [
    [1],
    [2,3],
    [1,2, 3, 4]
])
def test_box_l(l):
    proj  = projectors.proj_box(l=l)
    l = jnp.asarray(l)
    x = l + 1
    assert_array_equal(proj(x), x)
    x = l - 1
    assert_array_equal(proj(x), l)

@pytest.mark.parametrize("u", [
    [1],
    [2,3],
    [1,2, 3, 4]
])
def test_box_u(u):
    proj  = projectors.proj_box(u=u)
    u = jnp.asarray(u)
    x = u + 1
    assert_array_equal(proj(x), u)
    x = u - 1
    assert_array_equal(proj(x), x)


@pytest.mark.parametrize("u", [
    [1],
    [2,3],
    [1,2, 3, 4]
])
def test_box_lu(u):
    u = jnp.asarray(u)
    l = jnp.zeros_like(u)
    proj  = projectors.proj_box(l=l, u=u)
    x = u + 1
    assert_array_equal(proj(x), u)
    x = l - 1
    assert_array_equal(proj(x), l)
    x = (l + u) / 2
    assert_array_equal(proj(x), x)


@pytest.mark.parametrize("x, t", [
    [[1,1], 2],
    [[1], 1],
    [[1,2,3,4], 6],
])
def test_conic_inside(x, t):
    proj = projectors.proj_conic()
    x = jnp.append(jnp.asarray(x), t)
    assert_array_equal(proj(x), x)

@pytest.mark.parametrize("x, t", [
    [[1,1], 1],
    [[1], 0.1],
    [[1,2,3,4], 5],
])
def test_conic_outside(x, t):
    proj = projectors.proj_conic()
    x = jnp.append(jnp.asarray(x), t)
    v = proj(x)
    # once inside, if we project again, we will get same vector
    assert_allclose(proj(v), v, atol=atol, rtol=rtol)
