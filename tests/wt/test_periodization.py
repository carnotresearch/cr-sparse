from .setup import *



def test_signal_sizes():
    mode = 'periodization'
    name = 'haar'
    wavelets = ['haar', 'db2', 'db4', 'db8', 'coif1', 'coif3']
    for name in wavelets:
        wavelet = wt.build_wavelet(name)
        for n in range(2, 20, 3):
            if n < wavelet.dec_len : continue
            x = jnp.arange(n)
            ca, cd = wt.dwt(x, wavelet, mode=mode)
            n2 = (n + 1) // 2
            assert ca.size == n2
            assert cd.size == n2
            xr = wt.idwt(ca, cd, wavelet, mode)
            x = wt.make_even_shape(x)
            assert_allclose(x, xr, atol=atol, rtol=rtol)