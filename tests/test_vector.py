import pytest


from cr.sparse import *
import jax.numpy as jnp


from numpy.testing import (assert_almost_equal, assert_allclose, assert_,
                           assert_equal, assert_raises, assert_raises_regex,
                           assert_array_equal, assert_warns)

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


def test_vec_shift_right():
    n = 10
    x = jnp.arange(n)
    y = vec_shift_right(x)
    assert_allclose(y[1:], x[:-1])
    assert y[0] == 0

def test_vec_rotate_right():
    n = 10
    x = jnp.arange(n)
    y = vec_rotate_right(x)
    assert_allclose(y[1:], x[:-1])
    assert y[0] == x[-1]

def test_vec_shift_left():
    n = 10
    x = jnp.arange(n)
    y = vec_shift_left(x)
    assert_allclose(y[:-1], x[1:])
    assert y[-1] == 0

def test_vec_rotate_left():
    n = 10
    x = jnp.arange(n)
    y = vec_rotate_left(x)
    assert_allclose(y[:-1], x[1:])
    assert y[-1] == x[0]

def test_vec_shift_right_n():
    n = 10
    x = jnp.arange(n)
    y = vec_shift_right_n(x, 1)
    assert_allclose(y[1:], x[:-1])
    assert y[0] == 0

def test_vec_rotate_right_n():
    n = 10
    x = jnp.arange(n)
    y = vec_rotate_right_n(x, 1)
    assert_allclose(y[1:], x[:-1])
    assert y[0] == x[-1]

def test_vec_shift_left_n():
    n = 10
    x = jnp.arange(n)
    y = vec_shift_left_n(x, 1)
    assert_allclose(y[:-1], x[1:])
    assert y[-1] == 0

def test_vec_rotate_left_n():
    n = 10
    x = jnp.arange(n)
    y = vec_rotate_left_n(x, 1)
    assert_allclose(y[:-1], x[1:])
    assert y[-1] == x[0]


def test_vec_repeat_at_end():
    n = 10
    x = jnp.arange(n)
    p = 4
    y = vec_repeat_at_end(x, p)
    z = jnp.arange(p)
    assert_array_equal(y, jnp.concatenate((x, z)))

def test_vec_repeat_at_start():
    n = 10
    x = jnp.arange(n)
    p = 4
    y = vec_repeat_at_start(x, p)
    z = jnp.arange(n-p, n)
    assert_array_equal(y, jnp.concatenate((z, x)))

def test_vec_centered():
    n = 10
    x = jnp.arange(n)
    y = vec_centered(x, 8)
    assert_array_equal(y, x[1:-1])
    
