from proximal_setup import *

@pytest.mark.parametrize("x", [
    [3,4, 5, 0], 
    [[2,4], [3,5]]
])
def test_l2(x):
    prox = proximal_ops.prox_l2()
    assert_array_equal(prox.func(x), cnb.arr_l2norm(x))
    t = 1. 
    px = prox.prox_op(x, t)
    assert cnb.arr_l2norm(px) <= cnb.arr_l2norm(x)
    px2, v2 = prox.prox_vec_val(x, t)
    assert_array_equal(px, px2)
    assert_array_equal(v2, prox.func(px2))

