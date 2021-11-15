from .setup import *



def test_dwt_idwt_basic():
    x = jnp.array([3, 7, 1, 1, -2, 5, 4, 6])
    cA, cD = wt.dwt(x, 'db2')
    cA_expect = jnp.array([5.65685425, 7.39923721, 0.22414387, 3.33677403, 7.77817459])
    cD_expect = jnp.array([-2.44948974, -1.60368225, -4.44140056, -0.41361256,
                 1.22474487])
    assert_allclose(cA, cA_expect, atol=atol, rtol=rtol)
    assert_allclose(cD, cD_expect, atol=atol, rtol=rtol)

    x_roundtrip = wt.idwt(cA, cD, 'db2')
    assert_allclose(x_roundtrip, x, rtol=rtol)

    # mismatched dtypes OK
    x_roundtrip2 = wt.idwt(cA.astype(jnp.float64), cD.astype(jnp.float32),
                             'db2')
    assert_allclose(x_roundtrip2, x, rtol=rtol, atol=atol)
    assert_(x_roundtrip2.dtype == float_type)

def test_idwt_mixed_complex_dtype():
    x = jnp.arange(8).astype(float)
    x = x + 1j*x[::-1]
    cA, cD = wt.dwt(x, 'db2')

    x_roundtrip = wt.idwt(cA, cD, 'db2')
    assert_allclose(x_roundtrip, x, rtol=rtol, atol=atol)

    # mismatched dtypes OK
    x_roundtrip2 = wt.idwt(cA.astype(jnp.complex128), cD.astype(jnp.complex64),
                             'db2')
    assert_allclose(x_roundtrip2, x, rtol=rtol, atol=atol)
    assert_(x_roundtrip2.dtype == complex_type)



def test_dwt_idwt_dtypes():
    wavelet = wt.build_wavelet('haar')
    for dt_in, dt_out in zip(dtypes_in, dtypes_out):
        x = jnp.ones(4, dtype=dt_in)
        errmsg = f"wrong dtype returned for input: {dt_in}"

        cA, cD = wt.dwt(x, wavelet)
        assert_(cA.dtype == cD.dtype == dt_out, f"dwt: {errmsg}, expected: {dt_out}, actual: {cA.dtype}" )

        x_roundtrip = wt.idwt(cA, cD, wavelet)
        assert_(x_roundtrip.dtype == dt_out, "idwt: " + errmsg)


def test_dwt_idwt_basic_complex():
    x = jnp.asarray([3, 7, 1, 1, -2, 5, 4, 6])
    x = x + 0.5j*x
    cA, cD = wt.dwt(x, 'db2')
    cA_expect = jnp.array([5.65685425, 7.39923721, 0.22414387, 3.33677403,
                            7.77817459])
    cA_expect = cA_expect + 0.5j*cA_expect
    cD_expect = jnp.array([-2.44948974, -1.60368225, -4.44140056, -0.41361256,
                            1.22474487])
    cD_expect = cD_expect + 0.5j*cD_expect
    assert_allclose(cA, cA_expect, rtol=rtol, atol=atol)
    assert_allclose(cD, cD_expect, rtol=rtol, atol=atol)

    x_roundtrip = wt.idwt(cA, cD, 'db2')
    assert_allclose(x_roundtrip, x, rtol=rtol)


def test_dwt_idwt_partial_complex():
    x = jnp.asarray([3, 7, 1, 1, -2, 5, 4, 6])
    x = x + 0.5j*x

    cA, cD = wt.dwt(x, 'haar')
    cA_rec_expect = jnp.array([5.0+2.5j, 5.0+2.5j, 1.0+0.5j, 1.0+0.5j,
                              1.5+0.75j, 1.5+0.75j, 5.0+2.5j, 5.0+2.5j])
    cA_rec = wt.idwt(cA, None, 'haar')
    assert_allclose(cA_rec, cA_rec_expect, rtol=rtol, atol=atol)

    cD_rec_expect = jnp.array([-2.0-1.0j, 2.0+1.0j, 0.0+0.0j, 0.0+0.0j,
                              -3.5-1.75j, 3.5+1.75j, -1.0-0.5j, 1.0+0.5j])
    cD_rec = wt.idwt(None, cD, 'haar')
    assert_allclose(cD_rec, cD_rec_expect, rtol=rtol, atol=atol)

    assert_allclose(cA_rec + cD_rec, x, rtol=rtol, atol=atol)

def test_dwt_wavelet_kwd():
    x = jnp.array([3, 7, 1, 1, -2, 5, 4, 6])
    w = wt.build_wavelet('sym3')
    cA, cD = wt.dwt(x, wavelet=w, mode='constant')
    cA_expect = jnp.array([4.38354585, 3.80302657, 7.31813271, -0.58565539, 4.09727044,
                 7.81994027])
    cD_expect = jnp.array([-1.33068221, -2.78795192, -3.16825651, -0.67715519,
                 -0.09722957, -0.07045258])
    assert_allclose(cA, cA_expect, rtol=rtol, atol=atol)
    assert_allclose(cD, cD_expect, rtol=rtol, atol=atol)

def test_dwt_coeff_len():
    x = jnp.array([3, 7, 1, 1, -2, 5, 4, 6])
    w = wt.build_wavelet('sym3')
    ln_modes = [wt.dwt_coeff_len(len(x), w.dec_len, mode) for mode in
                wt.modes]

    expected_result = [6, ] * len(wt.modes)
    expected_result[wt.modes.index('periodization')] = 4

    assert_allclose(ln_modes, expected_result, rtol=rtol, atol=atol)
    ln_modes = [wt.dwt_coeff_len(len(x), w, mode) for mode in
                wt.modes]
    assert_allclose(ln_modes, expected_result, rtol=rtol, atol=atol)

def test_idwt_none_input():
    # None input equals arrays of zeros of the right length
    res1 = wt.idwt(jnp.array([1, 2, 0, 1]), None, 'db2', 'symmetric')
    res2 = wt.idwt(jnp.array([1, 2, 0, 1]), jnp.array([0, 0, 0, 0]), 'db2', 'symmetric')
    assert_allclose(res1, res2, rtol=1e-15, atol=1e-15)

    res1 = wt.idwt(None, jnp.array([1, 2, 0, 1]), 'db2', 'symmetric')
    res2 = wt.idwt(jnp.array([0, 0, 0, 0]), jnp.array([1, 2, 0, 1]), 'db2', 'symmetric')
    assert_allclose(res1, res2, rtol=1e-15, atol=1e-15)

    # Only one argument at a time can be None
    assert_raises(ValueError, wt.idwt, None, None, 'db2', 'symmetric')

def test_idwt_invalid_input():
    # Too short, min length is 4 for 'db4':
    assert_raises(ValueError, wt.idwt, jnp.array([1, 2, 4]), jnp.array([4, 1, 3]), 'db4', 'symmetric')


def test_dwt_single_axis():
    x = jnp.array([[3, 7, 1, 1],
         [-2, 5, 4, 6]])

    cA, cD = wt.dwt(x, 'db2', axis=-1)

    cA0, cD0 = wt.dwt(x[0], 'db2')
    cA1, cD1 = wt.dwt(x[1], 'db2')

    assert_allclose(cA[0], cA0)
    assert_allclose(cA[1], cA1)

    assert_allclose(cD[0], cD0)
    assert_allclose(cD[1], cD1)

def test_idwt_single_axis():
    x = jnp.array([[3, 7, 1, 1],
         [-2, 5, 4, 6]])

    x = np.asarray(x)
    x = x + 1j*x   # test with complex data
    cA, cD = wt.dwt(x, 'db2', axis=-1)

    x0 = wt.idwt(cA[0], cD[0], 'db2', axis=-1)
    x1 = wt.idwt(cA[1], cD[1], 'db2', axis=-1)

    assert_allclose(x[0], x0, atol=atol, rtol=rtol)
    assert_allclose(x[1], x1, atol=atol, rtol=rtol)


def test_dwt_axis_arg():
    x = jnp.array([[3, 7, 1, 1],
         [-2, 5, 4, 6]])

    cA_, cD_ = wt.dwt(x, 'db2', axis=-1)
    cA, cD = wt.dwt(x, 'db2', axis=1)

    assert_allclose(cA_, cA)
    assert_allclose(cD_, cD)


def test_idwt_axis_arg():
    x = jnp.array([[3, 7, 1, 1],
         [-2, 5, 4, 6]])

    cA, cD = wt.dwt(x, 'db2', axis=1)

    x_ = wt.idwt(cA, cD, 'db2', axis=-1)
    x = wt.idwt(cA, cD, 'db2', axis=1)

    assert_allclose(x_, x)

def test_dwt_idwt_axis_excess():
    x = jnp.array([[3, 7, 1, 1],
         [-2, 5, 4, 6]])
    # can't transform over axes that aren't there
    assert_raises(ValueError,
                  wt.dwt, x, 'db2', 'symmetric', axis=2)

    assert_raises(ValueError,
                  wt.idwt, jnp.array([1, 2, 4]), jnp.array([4, 1, 3]), 'db2', 'symmetric', axis=1)


