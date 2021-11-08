from .lop_setup import *

@pytest.mark.parametrize("in_shape,out_shape", [
    [(3,4), (2,6)],
    [(3,4,6), (6,4,3)],
])
def test_reshape(in_shape,out_shape):
    T = lop.reshape(in_shape, out_shape)
    T = lop.jit(T)
    x = jnp.ones(in_shape)
    y = T.times(x)
    yy = jnp.ones(out_shape)
    assert_array_equal(y, yy)
    z = T.trans(y)
    assert_array_equal(z, x)
