import pytest


from cr.sparse import *
import jax.numpy as jnp



def test_scalar():
    x = jnp.array(2)
    assert is_scalar(x)

def test_line_vec():
    x = jnp.array([2,3])
    assert is_line_vec(x)
    assert is_vec(x)


def test_row_vec():
    x = jnp.array([[2,3]])
    assert is_row_vec(x)
    assert is_vec(x)

def test_col_vec():
    x = jnp.array([[2],[3]])
    assert is_col_vec(x)
    assert is_vec(x)

def test_to_row_vec():
    x = jnp.array([2,3])
    assert is_line_vec(x)
    x = to_row_vec(x)
    assert is_row_vec(x)

def test_to_col_vec():
    x = jnp.array([2,3])
    assert is_line_vec(x)
    x = to_col_vec(x)
    assert is_col_vec(x)
