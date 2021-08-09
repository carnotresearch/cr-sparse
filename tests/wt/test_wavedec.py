from .setup import *


def test_compare_downcoef_coeffs():
    r = random.normal(keys[0], (16,))
    # compare downcoef against wavedec outputs
    for nlevels in [1, 2, 3]:
        for wavelet in wt.wavelist(kind='discrete'):
            wavelet = wt.build_wavelet(wavelet)
            if wavelet is not None:
                max_level = wt.dwt_max_level(r.size, wavelet.dec_len)
                if nlevels <= max_level:
                    a = wt.downcoef('a', r, wavelet, level=nlevels)
                    d = wt.downcoef('d', r, wavelet, level=nlevels)
                    coeffs = wt.wavedec(r, wavelet, level=nlevels)
                    assert_allclose(a, coeffs[0], rtol=rtol)
                    assert_allclose(d, coeffs[1], rtol=rtol)



def test_wavedec():
    x = [3, 7, 1, 1, -2, 5, 4, 6]
    db1 = wt.build_wavelet('db1')
    cA3, cD3, cD2, cD1 = wt.wavedec(x, db1)
    assert_almost_equal(cA3, [8.83883476], decimal=decimal_cmp)
    assert_almost_equal(cD3, [-0.35355339], decimal=decimal_cmp)
    assert_allclose(cD2, [4., -3.5], rtol=rtol, atol=atol)
    assert_allclose(cD1, [-2.82842712, 0, -4.94974747, -1.41421356], rtol=rtol, atol=atol)
    assert_(wt.dwt_max_level(len(x), db1) == 3)


