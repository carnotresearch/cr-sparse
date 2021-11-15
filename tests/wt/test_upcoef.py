from .setup import *

def test_upcoef_reconstruct():
    data = jnp.arange(6)
    a = wt.downcoef('a', data, 'haar')
    d = wt.downcoef('d', data, 'haar')
    coeffs  = (a, d)

    ra = wt.upcoef('a', a, 'haar')
    rd = wt.upcoef('d', d, 'haar')
    r = ra + rd
    assert jnp.allclose(data, r, rtol=rtol, atol=atol)


def test_upcoef_multilevel():
    r = random.normal(keys[0], (4,))
    nlevels = 3
    # calling with level=1 nlevels times
    a1 = r
    for i in range(nlevels):
        a1 = wt.upcoef('a', a1, 'haar', level=1)
    # call with level=nlevels once
    a3 = wt.upcoef('a', r, 'haar', level=nlevels)
    assert_allclose(a1, a3)


def test_upcoef_complex():
    r1 = random.normal(keys[0], (16,))
    r2 = random.normal(keys[1], (16,))
    r = r1 + 1j * r2
    nlevels = 3
    a = wt.upcoef('a', r, 'haar', level=nlevels)
    a_real = wt.upcoef('a', r.real, 'haar', level=nlevels)
    a_imag = wt.upcoef('a', r.imag, 'haar', level=nlevels)
    a_ref = a_real + 1j * a_imag
    assert_allclose(a, a_ref)

def test_upcoef_errs():
    # invalid part string (not 'a' or 'd')
    assert_raises(ValueError, wt.upcoef, 'f', jnp.ones(16), 'haar')


def test_updown_periodization():
    name = 'db4'
    wavelet = wt.build_wavelet(name)
    mode = 'periodization'
    signal = jnp.ones(128)
    ca, cd = wt.dwt(signal, wavelet, mode=mode)
    a = wt.upcoef('a', ca, wavelet, mode=mode)
    d = wt.upcoef('d', cd, wavelet, mode=mode)
    assert_allclose(a + d, signal, atol=atol, rtol=rtol)


def test_updown_symmetric():
    mode = 'symmetric'
    names = ['db4', 'db8', 'sym2', 'coif1']
    for name in names:
        wavelet = wt.build_wavelet(name)
        p = wavelet.dec_len
        skip = p -2
        for i in range(10, 50, 10):
            c = jnp.ones(i)
            ca = wt.downcoef('a', c, wavelet, mode=mode)
            cd = wt.downcoef('d', c, wavelet, mode=mode)
            a = wt.upcoef('a', ca, wavelet, mode=mode)
            d = wt.upcoef('d', cd, wavelet, mode=mode)
            r = a + d
            rec = r[skip:-skip]
            assert_allclose(rec, c, atol=atol, rtol=rtol)
