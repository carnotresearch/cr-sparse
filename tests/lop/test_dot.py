from .lop_setup import *


@pytest.mark.parametrize("v", [ 
    [1,2,4,5],
    [1,2+3j,4,5],
    [[1,2],[4,5]],
    [[1+2j,2],[4,5]],
    [[1+2j,2j],[4j,5+3j]],
])
def test_dot(v):
    v = jnp.asarray(v)
    T = lop.jit(lop.dot(v))
    x = random.normal(crs.KEYS[1], v.shape)
    z = crs.arr_rdot(v, x)
    assert_almost_equal(z, T.times(x))
    assert_allclose(v * z, T.trans(z))
    TH = lop.jit(lop.dot(v, adjoint=True))
    assert lop.rdot_test_complex(keys[2], T)
    assert_allclose(lop.to_matrix(TH), lop.to_adjoint_matrix(T))
    assert_allclose(lop.to_matrix(T), lop.to_adjoint_matrix(TH))
    #assert lop.rdot_test_complex(keys[3], TH)
