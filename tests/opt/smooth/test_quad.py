from smooth_setup import *


@pytest.mark.parametrize("x", [
    [3,4, 5, 0], 
    [[2,4], [3,5]]
])
def test_quad(x):
    f = smooth_quad_matrix()
    v = f.func(x)
    g = f.grad(x)
    g2, v2 = f.grad_val(x)
    assert_array_equal(g, g2)
    assert_array_equal(v, v2)

@pytest.mark.parametrize("x", [
    [3,4, 5, 0], 
])
def test_quad_p(x):
    n = len(x)
    P = jnp.eye(n)
    f = smooth_quad_matrix(P=P)
    v = f.func(x)
    g = f.grad(x)
    g2, v2 = f.grad_val(x)
    assert_array_equal(g, g2)
    assert_array_equal(v, v2)

@pytest.mark.parametrize("x", [
    [3,4, 5, 0], 
])
def test_quad_p_q(x):
    n = len(x)
    P = jnp.eye(n)
    q = jnp.zeros(n)
    f = smooth_quad_matrix(P=P, q=q)
    v = f.func(x)
    g = f.grad(x)
    g2, v2 = f.grad_val(x)
    assert_array_equal(g, g2)
    assert_array_equal(v, v2)


@pytest.mark.parametrize("x", [
    [3,4, 5, 0], 
])
def test_quad_p_q_r(x):
    n = len(x)
    P = jnp.eye(n)
    q = jnp.zeros(n)
    r = 0.
    f = smooth_quad_matrix(P=P, q=q, r=r)
    v = f.func(x)
    g = f.grad(x)
    g2, v2 = f.grad_val(x)
    assert_array_equal(g, g2)
    assert_array_equal(v, v2)
