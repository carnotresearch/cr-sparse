from .setup import *

def test_dwt_max_level():
    assert_(wt.dwt_max_level(16, 2) == 4)
    assert_(wt.dwt_max_level(16, 8) == 1)
    assert_(wt.dwt_max_level(16, 9) == 1)
    assert_(wt.dwt_max_level(16, 10) == 0)
    assert_(wt.dwt_max_level(16, np.int8(10)) == 0)
    assert_(wt.dwt_max_level(16, 10.) == 0)
    assert_(wt.dwt_max_level(16, 18) == 0)

    # accepts discrete Wavelet object or string as well
    assert_(wt.dwt_max_level(32, wt.build_wavelet('sym5')) == 1)
    assert_(wt.dwt_max_level(32, 'sym5') == 1)

    # string input that is not a discrete wavelet
    assert_raises(TypeError, wt.dwt_max_level, 16, 'mexh')

    # filter_len must be an integer >= 2
    assert_raises(ValueError, wt.dwt_max_level, 16, 1)
    assert_raises(ValueError, wt.dwt_max_level, 16, -1)
    assert_raises(ValueError, wt.dwt_max_level, 16, 3.3)


