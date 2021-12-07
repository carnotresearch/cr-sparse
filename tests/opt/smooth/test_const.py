from smooth_setup import *


@pytest.mark.parametrize("x", [
    [3,4, 5, 0], 
    [[2,4], [3,5]]
])
def test_constant(x):
    f = smooth_constant()
    assert_array_equal(f.func(x), 0.)
    g = f.grad(x)
    assert_array_equal(g, 0.)
    g2, v2 = f.grad_val(x)
    assert_array_equal(g, g2)
    assert_array_equal(v2, f.func(x))