

from .setup import *

from cr.sparse.wt.wlab import *

def test_iconv():
    f  = jnp.array([1., 1.])
    x  = random.normal(keys[0], (4,))
    iconv(f, x)

def test_aconv():
    f  = jnp.array([1., 1.])
    x  = random.normal(keys[0], (4,))
    aconv(f, x)


def test_mirror():
    f  = jnp.array([1., 1.])
    h = mirror_filter(f)


def test_up_sample():
    x  = random.normal(keys[0], (4,))
    y = up_sample(x, 2)

def test_lo_pass_down_sample():
    h  = jnp.array([1., 1.])
    x  = random.normal(keys[0], (4,))
    lo_pass_down_sample(h, x)

def test_hi_pass_down_sample():
    h  = jnp.array([-1., 1.])
    x  = random.normal(keys[0], (4,))
    hi_pass_down_sample(h, x)

def test_up_sample_lo_pass():
    h  = jnp.array([1., 1.])
    x  = random.normal(keys[0], (4,))
    up_sample_lo_pass(h, x)

def test_up_sample_hi_pass():
    h  = jnp.array([-1., 1.])
    x  = random.normal(keys[0], (4,))
    up_sample_hi_pass(h, x)

def test_downsampling_convolution_periodization():
    h  = jnp.array([1., 1.])
    x  = random.normal(keys[0], (4,))
    downsampling_convolution_periodization(h,x)

def test_filter_coefs():
    haar()
    db4()
    db6()
    db8()
    db10()
    db12()
    db14()
    db16()
    db18()
    db20()
    baylkin()
    coif1()
    coif2()
    coif3()
    coif4()
    coif5()
    symm4()
    symm5()
    symm6()
    symm7()
    symm8()
    symm9()
    symm10()
    vaidyanathan()

def test_wavelet_function():
    wavelet_function(db4(), 0, 0, 16)

def test_scaling_function():
    scaling_function(db4(), 0, 0, 16)


def test_forward_periodized_orthogonal():
    qmf = haar()
    x  = random.normal(keys[0], (4,))
    w = forward_periodized_orthogonal_jit(qmf, x)

def test_inverse_periodized_orthogonal():
    qmf = haar()
    w  = random.normal(keys[0], (4,))
    x = inverse_periodized_orthogonal_jit(qmf, w)
