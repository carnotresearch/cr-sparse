from smooth_setup import *


@pytest.mark.parametrize("x", [
    [[2,4], [3,5]]
])
def test_logdet(x):
    f = smooth_logdet()
    v = f.func(x)
    g = f.grad(x)
    g2, v2 = f.grad_val(x)
    assert_array_equal(g, g2)
    assert_array_equal(v, v2)

@pytest.mark.parametrize("x", [
    [[2,4], [3,5]]
])
def test_logdet_c(x):
    x = jnp.asarray(x)
    n = x.shape[0]
    C = jnp.eye(n)
    f = smooth_logdet(C=C)
    v = f.func(x)
    g = f.grad(x)
    g2, v2 = f.grad_val(x)
    assert_array_equal(g, g2)
    assert_array_equal(v, v2)
