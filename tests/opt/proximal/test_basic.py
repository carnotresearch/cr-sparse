from proximal_setup import *


@pytest.mark.parametrize("x", [
    [3,4, 5, 0], 
    [[2,4], [3,5]]
])
def test_zero(x):
    prox = proximal_ops.prox_zero()
    assert_array_equal(prox.func(x), 0)
    t = 1. 
    assert_array_equal(prox.prox_op(x, t), x)
    px, v = prox.prox_vec_val(x, t)
    assert_array_equal(px, x)
    assert_array_equal(v, 0)
