from .setup import *

def test_wavelist():
    for name in wt.wavelist(family='coif'):
        assert_(name.startswith('coif'))

    #assert_('cgau7' in wt.wavelist(kind='continuous'))
    assert_('sym20' in wt.wavelist(kind='discrete'))
    assert_(len(wt.wavelist(kind='continuous')) +
            len(wt.wavelist(kind='discrete')) ==
            len(wt.wavelist(kind='all')))

    assert_raises(ValueError, wt.wavelist, kind='foobar')

def test_wavelet_errormsgs():
    assert_raises(ValueError, wt.build_wavelet, 'gaus1')
    assert_raises(ValueError, wt.build_wavelet, 'cmord')
