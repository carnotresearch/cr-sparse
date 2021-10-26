from math import sqrt

from cr.sparse import *
import jax.numpy as jnp

import pytest

# Integers
X = jnp.array([[1, 2],
             [3, 4]])

# Floats
X2 = jnp.array([[1, 2],
             [3, 4]], dtype=jnp.float32)

def check_equal(actual, expected):
    print(actual, expected)
    x = jnp.equal(actual, expected)
    print(x)
    x = jnp.all(x)
    print(x)
    assert x


def check_approx_equal(actual, expected, abs_err=1e-6):
    result = jnp.allclose(actual, expected, atol=abs_err)
    print(result)
    assert result


def test_l1_norm_cw():
    check_equal(norms_l1_cw(X), jnp.array([4, 6]))

def test_l1_norm_rw():
    check_equal(norms_l1_rw(X), jnp.array([3, 7]))


def test_l2_norm_cw():
    check_approx_equal(norms_l2_cw(X2), jnp.array([sqrt(10), sqrt(20)]))

def test_l2_norm_rw():
    check_approx_equal(norms_l2_rw(X2), jnp.array([sqrt(5), sqrt(25)]))
 

def test_linf_norm_cw():
    check_equal(norms_linf_cw(X), jnp.array([3, 4]))

def test_linf_norm_rw():
    check_equal(norms_linf_rw(X), jnp.array([2, 4]))


def test_normalize_l1_cw():
    Y = normalize_l1_cw(X2)
    check_equal(norms_l1_cw(Y), jnp.array([1., 1.]))

def test_normalize_l1_rw():
    Y = normalize_l1_rw(X2)
    check_equal(norms_l1_rw(Y), jnp.array([1., 1.]))


def test_normalize_l2_cw():
    Y = normalize_l2_cw(X2)
    check_approx_equal(norms_l2_cw(Y), jnp.array([1., 1.]))

def test_normalize_l2_rw():
    Y = normalize_l2_rw(X2)
    check_approx_equal(norms_l2_rw(Y), jnp.array([1., 1.]))


def test_norm_l1():
    n = 32
    x = jnp.arange(n)
    assert jnp.sum(x) == norm_l1(x)

def test_sqr_norm_l2():
    n = 32
    x = jnp.arange(n)
    assert jnp.sum(x**2) == sqr_norm_l2(x)

def test_norm_l2():
    n = 32
    x = jnp.arange(n)
    assert jnp.sum(x**2) == norm_l2(x)**2

def test_norm_inf():
    n = 32
    x = jnp.arange(n)
    assert n-1 == norm_linf(x)