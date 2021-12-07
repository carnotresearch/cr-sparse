from indicators_setup import *


def test_zero():
    f = indicator_zero()
    assert_array_equal(f(0), 0)
    assert_array_equal(f(1), jnp.inf)
    assert_array_equal(f([0, 0]), 0)
    assert_array_equal(f([1, 0]), jnp.inf)


def test_singleton_1():
    f = indicator_singleton(1)
    assert_array_equal(f(1), 0)
    assert_array_equal(f(2), jnp.inf)
    assert_array_equal(f([1, 1]), 0)
    assert_array_equal(f([1, 0]), jnp.inf)

def test_singleton_2():
    f = indicator_singleton([1, 1])
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
    f = indicator_affine(A, b)
    assert_array_equal(f(x), 0)
 

def test_affine_2():
    # This A simply keeps the first component of x
    A = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    f = indicator_affine(A)
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
    f = indicator_affine(A, [b, 0, 0])
    assert_array_equal(f([b, 1, 1, -1]), 0)
    assert_array_equal(f([b, 0, 1, -10]), 0)
    assert_array_equal(f([b+1, 0, 1, -10]), jnp.inf)
    assert_array_equal(f([b-1, 0, 0, 0]), jnp.inf)


@pytest.mark.parametrize("l,u", [
    [0, None],
    [None, 1],
    [0, 4],
    [[0,0], [4, 4]],
])
def test_box(l, u):
    box = indicator_box(l, u)
    if l is not None:
        assert_array_equal(box(l), 0)
        l = jnp.asarray(l)
        assert_array_equal(box(l+1), 0)
        assert_array_equal(box(l-1), jnp.inf)
    if u is not None:
        assert_array_equal(box(u), 0)
        u = jnp.asarray(u)
        assert_array_equal(box(u-1), 0)
        assert_array_equal(box(u+1), jnp.inf)
    if l is not None and u is not None:
        x = (l + u) / 2
        assert_array_equal(box(x), 0)


@pytest.mark.parametrize("l,u,shape", [
    [0, None, (4,1)],
    [None, 1, (4,2)],
    [0, 4, (1,3)],
    [[0,0], [3, 4], (2,)],
    [[0,0], [3, 4], (2,2)],
])
def test_box_affine(l,u,shape):
    a = jnp.ones(shape)
    n = a.size
    box = indicator_box_affine(l, u, a, 1)
    x = jnp.ones(shape)
    x = x / crs.arr_rdot(a, x)
    assert_array_equal(box(x), 0)
    if l is not None:
        l = jnp.asarray(l)
        x = jnp.broadcast_to(l -1, shape)
        assert_array_equal(box(x), jnp.inf)
    if u is not None:
        u = jnp.asarray(u)
        x = jnp.broadcast_to(u +1, shape)
        assert_array_equal(box(x), jnp.inf)


@pytest.mark.parametrize("x, t", [
    [[1,1], 2],
    [[1], 1],
    [[1,2,3,4], 6],
])
def test_conic_inside(x, t):
    f = indicator_conic()
    x = jnp.append(jnp.asarray(x), t)
    assert_array_equal(f(x), 0)

@pytest.mark.parametrize("x, t", [
    [[1,1], 1],
    [[1], 0.1],
    [[1,2,3,4], 5],
])
def test_conic_outside(x, t):
    f = indicator_conic()
    x = jnp.append(jnp.asarray(x), t)
    assert_array_equal(f(x), jnp.inf)