import pytest


import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose, assert_,
                           assert_equal, assert_raises, assert_raises_regex,
                           assert_array_equal, assert_warns)


import cr.sparse as crs
import jax.numpy as jnp


def test_promote_arg_dtypes():
    res = crs.promote_arg_dtypes(jnp.array(1), jnp.array(2))
    expected = jnp.array([1.0, 2.0])
    assert jnp.array_equal(res, expected)
    assert jnp.array_equal(crs.promote_arg_dtypes(jnp.array(1)), jnp.array(1.))
    crs.promote_arg_dtypes(jnp.array(1), jnp.array(2.))

def test_canonicalize_dtype():
    crs.canonicalize_dtype(None)
    crs.canonicalize_dtype(jnp.int32)


def test_is_cpu():
    assert_equal(crs.is_cpu(), crs.platform == 'cpu')

def test_is_gpu():
    assert_equal(crs.is_gpu(), crs.platform == 'gpu')

def test_is_tpu():
    assert_equal(crs.is_tpu(), crs.platform == 'tpu')

def test_check_shapes_are_equal():
    z = jnp.zeros(4)
    o = jnp.ones(4)
    crs.check_shapes_are_equal(z, o)
    o = jnp.ones(5)
    with assert_raises(ValueError):
        crs.check_shapes_are_equal(z, o)

def test_promote_to_complex():
    z = jnp.zeros(4)
    z = crs.promote_to_complex(z)
    assert z.dtype == np.complex64

def test_promote_to_real():
    z = jnp.zeros(4, dtype=int)
    z = crs.promote_to_real(z)
    assert z.dtype == np.float32


def test_nbytes_live_buffers():
    nbytes = crs.nbytes_live_buffers()
    assert nbytes > 0