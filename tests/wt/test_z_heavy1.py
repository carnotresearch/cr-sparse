from .setup import *

def test_waverec_all_wavelets_modes():
    # test 2D case using all wavelets and modes
    r = random.normal(keys[0], (64,))
    for wavelet in discrete_wavelets:
        for mode in wt.modes:
            coeffs = wt.wavedec(r, wavelet, mode=mode)
            assert_allclose(wt.waverec(coeffs, wavelet, mode=mode),
                            r, rtol=rtol, atol=atol)
