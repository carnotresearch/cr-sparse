from .setup import *

def test_waverec_invalid_inputs():
    # input must be list or tuple
    assert_raises(ValueError, wt.waverec, np.ones(8), 'haar')

    # input list cannot be empty
    assert_raises(ValueError, wt.waverec, [], 'haar')

    # 'array_to_coeffs must specify 'output_format' to perform waverec
    x = [3, 7, 1, 1, -2, 5, 4, 6]
    coeffs = wt.wavedec(x, 'db1')
    # arr, coeff_slices = wt.coeffs_to_array(coeffs)
    # coeffs_from_arr = wt.array_to_coeffs(arr, coeff_slices)
    # message = "Unexpected detail coefficient type"
    # assert_raises_regex(ValueError, message, wt.waverec, coeffs_from_arr,
    #                     'haar')


def test_waverec_accuracies():
    x0 = random.normal(keys[0], (8,))
    x1 = random.normal(keys[1], (8,))
    for dtype in [jnp.float16, jnp.float32, jnp.float64]:
        x = x0.astype(dtype)
        coeffs = wt.wavedec(x, 'db1')
        x_rec = wt.waverec(coeffs, 'db1')
        assert_allclose(x, x_rec, atol=atol, rtol=rtol)
    for dtype in [jnp.complex64, jnp.complex128]:
        x = lax.complex(x0, x1).astype(dtype)
        coeffs = wt.wavedec(x, 'db1')
        x_rec = wt.waverec(coeffs, 'db1')
        assert_allclose(x, x_rec, atol=atol, rtol=rtol)


def test_waverec_none():
    x = [3, 7, 1, 1, -2, 5, 4, 6]
    coeffs = wt.wavedec(x, 'db1')

    # set some coefficients to None
    coeffs[2] = None
    coeffs[0] = None
    assert_(wt.waverec(coeffs, 'db1').size, len(x))

#@pytest.mark.skip(reason="odd lengths not supported yet")
def test_waverec_odd_length():
    x = [3, 7, 1, 1, -2, 5]
    coeffs = wt.wavedec(x, 'db1')
    assert_allclose(wt.waverec(coeffs, 'db1'), x, rtol=rtol)

def test_waverec_complex():
    x = jnp.array([3, 7, 1, 1, -2, 5, 4, 6])
    x = x + 1j
    coeffs = wt.wavedec(x, 'db1')
    assert_allclose(wt.waverec(coeffs, 'db1'), x, rtol=rtol)


