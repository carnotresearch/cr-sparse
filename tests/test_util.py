import pytest


from cr.sparse import *
import jax.numpy as jnp


def test_promote_arg_dtypes():
    res = promote_arg_dtypes(jnp.array(1), jnp.array(2))
    expected = jnp.array([1.0, 2.0])
    assert jnp.array_equal(res, expected)
    assert jnp.array_equal(promote_arg_dtypes(jnp.array(1)), jnp.array(1.))
    promote_arg_dtypes(jnp.array(1), jnp.array(2.))

def test_canonicalize_dtype():
    canonicalize_dtype(None)
    canonicalize_dtype(jnp.int32)
