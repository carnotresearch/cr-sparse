from proximal_setup import *

@pytest.mark.parametrize("x", [
    [3,4, 5, 0], 
    [[2,4], [3,5]]
])
def test_l1(x):
    prox = proximal_ops.prox_l1()
    assert_array_equal(prox.func(x), crs.arr_l1norm(x))
    t = 1. 
    px = prox.prox_op(x, t)
    assert crs.arr_l1norm(px) <= crs.arr_l1norm(x)
    px2, v2 = prox.prox_vec_val(x, t)
    assert_array_equal(px, px2)
    assert_array_equal(v2, prox.func(px2))


@pytest.mark.parametrize("x", [
    [3,4, 5, 0], 
    [[2,4], [3,5]]
])
def test_l1_pos(x):
    prox = proximal_ops.prox_l1_pos()
    assert_array_equal(prox.func(x), crs.arr_l1norm(x))
    t = 1. 
    px = prox.prox_op(x, t)
    assert crs.arr_l1norm(px) <= crs.arr_l1norm(x)
    px2, v2 = prox.prox_vec_val(x, t)
    assert_array_equal(px, px2)
    assert_array_equal(v2, prox.func(px2))

