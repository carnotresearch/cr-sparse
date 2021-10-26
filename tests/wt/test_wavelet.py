from .setup import *

import cr.sparse._src.wt.wavelet as wt_int

@pytest.mark.parametrize("name", ['haar', 'db2', 'sym4', 'coif2', 'bior1.1', 'rbio1.1', 'dmey'])
def test_build_wavelet(name):
    wavelet = wt.build_wavelet(name)
    assert wavelet
    s = str(wavelet)
    bank = wavelet.filter_bank
    assert len(bank) == 4
    inverse_bank = wavelet.inverse_filter_bank
    assert len(inverse_bank) == 4
    assert_array_equal(bank[0], inverse_bank[2][::-1])
    assert_array_equal(bank[1], inverse_bank[3][::-1])
    assert_array_equal(bank[2], inverse_bank[0][::-1])
    assert_array_equal(bank[3], inverse_bank[1][::-1])
    wavefun = wavelet.wavefun(2)
    assert wavefun is not None
    integration = wt.integrate_wavelet(wavelet, 2)
    assert integration is not None

@pytest.mark.parametrize("name,order", [ (wt.FAMILY.CMOR, 4),
    (wt.FAMILY.DB, 100), (wt.FAMILY.SYM, 400), (wt.FAMILY.COIF, 500), 
    (wt.FAMILY.BIOR, 120), (wt.FAMILY.RBIO, 340), ])
def test_build_wavelet_fail(name, order):
    wavelet = wt.build_discrete_wavelet(name, order)
    assert wavelet is None

def test_fake():
    wavelet = wt.DiscreteWavelet()
    with assert_raises(NotImplementedError):
        wavelet.wavefun()


def test_qmf():
    h = jnp.ones(4)
    g = wt_int.qmf(h)
    hh = wt_int.qmf(g)
    assert_array_equal(h, -hh)

def test_orthogonal_filter_bank():
    h = jnp.ones(4)
    bank = wt_int.orthogonal_filter_bank(h)
    dec_lo, dec_hi, rec_lo, rec_hi = bank
    assert_array_equal(rec_hi, wt_int.qmf(rec_lo))

def test_filter_bank_():
    h = jnp.ones(4)
    bank = wt_int.filter_bank_(h)
    dec_lo, dec_hi, rec_lo, rec_hi = bank
    assert_array_equal(rec_hi, wt_int.qmf(rec_lo))


def test_mirror():
    h = jnp.ones(4)
    h_m = wt_int.mirror(h)
    h_mm = wt_int.mirror(h_m)
    assert_array_equal(h, h_mm)

def test_negate_evens():
    h = jnp.ones(4)
    h_m = wt_int.negate_evens(h)
    h_mm = wt_int.negate_evens(h_m)
    assert_array_equal(h, h_mm)

def test_negate_odds():
    h = jnp.ones(4)
    h_m = wt_int.negate_odds(h)
    h_mm = wt_int.negate_odds(h_m)
    assert_array_equal(h, h_mm)

@pytest.mark.parametrize("n,m", [ (1, 1), (2, 2), (2, 4),
    (3, 1), (3,9), (4, 4), (5, 5), (6, 8)])
def test_bior_index(n, m):
    idx, max =  wt_int.bior_index(n, m)
    assert idx >= 0
    assert max > 0

def test_bior_index_invalid():
    idx, max =  wt_int.bior_index(7, 0)
    assert idx is None
    assert max is None

@pytest.mark.parametrize("name", ['mexh', 'cmor1.5-2.0'])
def test_build_continuous_wavelet(name):
    wavelet = wt.build_wavelet(name)
    assert wavelet
    s = str(wavelet)
    wavefun = wavelet.wavefun(level=2)
    assert wavefun is not None
    assert wavelet.domain > 0
    cf = wt.central_frequency(wavelet)
    scales = jnp.array([1., 2.])
    frequencies = wt.scale2frequency(wavelet, scales)
    integrated = wt.integrate_wavelet(wavelet, precision=2)


def test_to_wavelet():
    wavelet = wt.to_wavelet('haar')
    wavelet2 = wt.to_wavelet(wavelet)
    assert wavelet2 is wavelet
    with assert_raises(ValueError):
        wt.to_wavelet(None)


@pytest.mark.parametrize("name,family,order,valid", [
    ('gaus', wt.FAMILY.GAUS, 2, True), 
    ('gaus', wt.FAMILY.GAUS, 3, True), 
    ('gaus', wt.FAMILY.GAUS, 10, False), 
    ('morl', wt.FAMILY.MORL, 0, True),
    ('cgau', wt.FAMILY.CGAU, 2, True), 
    ('cgau', wt.FAMILY.CGAU, 3, True), 
    ('cgau', wt.FAMILY.CGAU, 10, False), 
    ('shan', wt.FAMILY.SHAN, 0, True),
    ('fbsp40.20', wt.FAMILY.FBSP, 0, True),
    ])
def test_unsupported_continuous_wavelets(name, family, order, valid):
    wavelet = wt.build_continuous_wavelet(name, family, order)
    if valid:
        assert wavelet is not None
    else:
        assert wavelet is None