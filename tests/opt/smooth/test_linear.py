from smooth_setup import *


@pytest.mark.parametrize("x", [
    [3,4, 5, 0], 
    [[2,4], [3,5]]
])
def test_linear(x):
    x = jnp.asarray(x)
    C = jnp.ones_like(x)
    f = smooth_linear(C)
    v = f.func(x)
    g = f.grad(x)
    g2, v2 = f.grad_val(x)
    assert_array_equal(g, g2)
    assert_array_equal(v, v2)
