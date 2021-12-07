from indicators_setup import *

# L2 Balls

@pytest.mark.parametrize("x,q,v", [
    [[3,4], 5, 0], 
    [[3,4], 4, jnp.inf], 
])
def test_l2_ball(x, q, v):
    ball = indicators.indicator_l2_ball(q)
    assert_array_equal(ball(x), v)


@pytest.mark.parametrize("x,q,b,v", [
    [[3,4], 5, 0, 0], 
    [[3,4], 4, 0, jnp.inf], 
    [[3,4], 5, [0,0], 0], 
    [[3,4], 4, [0,0], jnp.inf], 
    [[3,4], 5, [1,1], 0], 
    [[4,5], 4, [1,1], jnp.inf], 
])
def test_l2_ball_b(x, q, b, v):
    ball = indicators.indicator_l2_ball(q, b=b)
    assert_array_equal(ball(x), v)


@pytest.mark.parametrize("x,q,b,v", [
    [[3,4], 5, 0, 0], 
    [[3,4], 4, 0, jnp.inf], 
    [[3,4], 5, [0,0], 0], 
    [[3,4], 4, [0,0], jnp.inf], 
    [[3,4], 5, [1,1], 0], 
    [[4,5], 4, [1,1], jnp.inf], 
])
def test_l2_ball_b_a(x, q, b, v):
    n = len(x)
    A = jnp.eye(n)
    ball = indicators.indicator_l2_ball(q, b=b, A=A)
    assert_array_equal(ball(x), v)


# L1 Balls


@pytest.mark.parametrize("x,q,v", [
    [[3,4], 7, 0], 
    [[3,4], 4, jnp.inf], 
])
def test_l1_ball(x, q, v):
    ball = indicators.indicator_l1_ball(q)
    assert_array_equal(ball(x), v)


@pytest.mark.parametrize("x,q,b,v", [
    [[3,4], 7, 0, 0], 
    [[3,4], 4, 0, jnp.inf], 
    [[3,4], 7, [0,0], 0], 
    [[3,4], 4, [0,0], jnp.inf], 
    [[3,4], 5, [1,1], 0], 
    [[4,5], 4, [1,1], jnp.inf], 
])
def test_l1_ball_b(x, q, b, v):
    ball = indicators.indicator_l1_ball(q, b=b)
    assert_array_equal(ball(x), v)


@pytest.mark.parametrize("x,q,b,v", [
    [[3,4], 7, 0, 0], 
    [[3,4], 4, 0, jnp.inf], 
    [[3,4], 7, [0,0], 0], 
    [[3,4], 4, [0,0], jnp.inf], 
    [[3,4], 5, [1,1], 0], 
    [[4,5], 4, [1,1], jnp.inf], 
])
def test_l1_ball_b_a(x, q, b, v):
    n = len(x)
    A = jnp.eye(n)
    ball = indicators.indicator_l1_ball(q, b=b, A=A)
    assert_array_equal(ball(x), v)

