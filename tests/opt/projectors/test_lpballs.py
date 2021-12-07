from projectors_setup import *

# L2 Balls

@pytest.mark.parametrize("x,q,outside", [
    [[3,4], 5, 0], 
    [[3,4], 4, 1], 
])
def test_l2_ball(x, q, outside):
    ball = projectors.proj_l2_ball(q)
    v = ball(x)
    if outside:
        # second projection should not change it
        assert_array_equal(ball(v), v)
    else:
        # projection should not change it
        assert_array_equal(v, x)

@pytest.mark.parametrize("x,q,b,outside", [
    [[3,4], 5, 0, 0], 
    [[3,4], 4, 0, 1], 
    [[3,4], 5, [0,0], 0], 
    [[3,4], 4, [0,0], 1], 
    [[3,4], 5, [1,1], 0], 
    [[4,5], 4, [1,1], 1], 
])
def test_l2_ball_b(x, q, b, outside):
    ball = projectors.proj_l2_ball(q, b=b)
    v = ball(x)
    if outside:
        # second projection should not change it
        assert_array_equal(ball(v), v)
    else:
        # projection should not change it
        assert_array_equal(v, x)


# @pytest.mark.parametrize("x,q,b,outside", [
#     [[3,4], 5, 0, 0], 
#     [[3,4], 4, 0, 1], 
#     [[3,4], 5, [0,0], 0], 
#     [[3,4], 4, [0,0], 1], 
#     [[3,4], 5, [1,1], 0], 
#     [[4,5], 4, [1,1], 1], 
# ])
# def test_l2_ball_b_a(x, q, b, outside):
#     n = len(x)
#     A = jnp.eye(n)
#     ball = projectors.proj_l2_ball(q, b=b, A=A)
#     v = ball(x)
#     if outside:
#         # second projection should not change it
#         assert_array_equal(ball(v), outside)
#     else:
#         # projection should not change it
#         assert_array_equal(v, x)


# # L1 Balls


@pytest.mark.parametrize("x,q,outside", [
    [[3,4], 7, 0], 
    [[3,4], 4, 1], 
])
def test_l1_ball(x, q, outside):
    ball = projectors.proj_l1_ball(q)
    v = ball(x)
    if outside:
        # second projection should not change it
        assert_array_equal(ball(v), v)
    else:
        # projection should not change it
        assert_array_equal(v, x)


@pytest.mark.parametrize("x,q,b,outside", [
    [[3,4], 7, 0, 0], 
    [[3,4], 4, 0, 1], 
    [[3,4], 7, [0,0], 0], 
    [[3,4], 4, [0,0], 1], 
    [[3,4], 5, [1,1], 0], 
    [[4,5], 4, [1,1], 1], 
])
def test_l1_ball_b(x, q, b, outside):
    ball = projectors.proj_l1_ball(q, b=b)
    v = ball(x)
    if outside:
        # second projection should not change it
        assert_array_equal(ball(v), v)
    else:
        # projection should not change it
        assert_array_equal(v, x)



