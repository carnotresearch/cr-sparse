from .setup import *



def test_downcoef_multilevel():
    r = random.normal(keys[0], (16,))
    nlevels = 3
    # calling with level=1 nlevels times
    a1 = r
    for i in range(nlevels):
        a1 = wt.downcoef('a', a1, 'haar', level=1)
    # call with level=nlevels once
    a3 = wt.downcoef('a', r, 'haar', level=nlevels)
    assert jnp.allclose(a1, a3)


def test_downcoef_complex():
    r1 = random.normal(keys[0], (16,))
    r2 = random.normal(keys[1], (16,))
    r = r1 + 1j * r2
    nlevels = 3
    a = wt.downcoef('a', r, 'haar', level=nlevels)
    a_real = wt.downcoef('a', r.real, 'haar', level=nlevels)
    a_imag = wt.downcoef('a', r.imag, 'haar', level=nlevels)
    a_ref = a_real + 1j * a_imag
    assert jnp.allclose(a, a_ref)


def test_downcoef_errs():
    # invalid part string (not 'a' or 'd')
    assert_raises(ValueError, wt.downcoef, 'f', jnp.ones(16), 'haar')
