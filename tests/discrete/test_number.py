import pytest


from cr.sparse import *
import jax.numpy as jnp

def test_numbers():
    assert is_integer(10)
    assert is_positive_integer(10)
    assert is_negative_integer(-10)
    assert is_odd(-3)
    assert is_even(40)
    assert is_odd_natural(5)
    assert is_even_natural(6)
    assert is_power_of_2(1024)
    assert is_perfect_square(81)
    a,b  = integer_factors_close_to_sqr_root(20)
    assert a == 4
    assert b == 5
    a,b  = integer_factors_close_to_sqr_root(25)
    assert a == 5
    assert b == 5
    a,b  = integer_factors_close_to_sqr_root(90)
    assert a == 9
    assert b == 10
    a,b  = integer_factors_close_to_sqr_root(77)
    assert a == 7
    assert b == 11

