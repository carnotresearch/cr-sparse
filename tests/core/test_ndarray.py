from core_setup import *


def test_arr_largest_index():
    x = jnp.array([1, -2, -3, 2])
    assert_equal(crs.arr_largest_index(x), (jnp.array([2]), ))
    x = jnp.reshape(x, (2,2))
    idx = crs.arr_largest_index(x)
    print(idx)
    assert_equal(idx, (jnp.array([1]), jnp.array([0])))


def test_arr_l2norm():
    x = jnp.zeros(10)
    assert_allclose(crs.arr_l2norm(x), 0.)

def test_arr_l2norm_sqr():
    x = jnp.zeros(10)
    assert_allclose(crs.arr_l2norm_sqr(x), 0.)

def test_arr_vdot():
    x = jnp.zeros(10)
    assert_allclose(crs.arr_vdot(x, x), 0.)
