from sls_setup import *

def test_identity_func():
    x = jnp.array([1, 2])
    assert_allclose(sls.identity_func(x), x)


def test_identity_op():
    x = jnp.array([1, 2])
    assert_allclose(sls.identity_op.times(x), x)
    assert_allclose(sls.identity_op.trans(x), x)


def test_default_threshold():
    x = jnp.array([1, 2, 3, 4, 5])
    y = sls.default_threshold(0, x)
    tau = 1.8
    z = jnp.maximum(jnp.abs(x) - tau, 0.)
    assert_allclose(z, y)

def test_default_threshold_cmplx():
    x = jnp.array([1, 2, 3, 4, 5]) + 0j
    y = sls.default_threshold(0, x)
    tau = 1.8
    z = jnp.maximum(jnp.abs(x) - tau, 0.)
    assert_allclose(z, y)
