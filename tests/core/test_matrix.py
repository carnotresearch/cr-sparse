import pytest

from cr.sparse import *

import jax.numpy as jnp


def test_is_symmetric():
    x = jnp.array([1,2])
    assert not is_symmetric(x)
    A = jnp.array([[1,2], [2, 1]])
    assert is_symmetric(A)

def test_is_hermitian():
    x = jnp.array([1,2])
    assert not is_hermitian(x)
    A = jnp.array([[1,2], [2, 1]])
    assert is_hermitian(A)

def test_is_positive_definite():
    x = jnp.array([1,2])
    assert not is_positive_definite(x)
    A = jnp.array([[1.,0], [0, 1]])
    if is_cpu():
        assert is_positive_definite(A)



