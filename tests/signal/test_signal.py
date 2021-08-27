import pytest


from cr.sparse import *
import jax.numpy as jnp
from jax import random

from numpy.testing import (assert_almost_equal, assert_allclose, assert_,
                           assert_equal, assert_raises, assert_raises_regex,
                           assert_array_equal, assert_warns)


def test_find_first_signal_with_energy_le_rw():
    X = jnp.eye(10)
    X = X.at[5,5].set(.5)
    index = find_first_signal_with_energy_le_rw(X, 0.3)
    assert index == 5
    index = find_first_signal_with_energy_le_rw(X, 0.2)
    assert index == -1

def test_find_first_signal_with_energy_le_cw():
    X = jnp.eye(10)
    X = X.at[5,5].set(.5)
    index = find_first_signal_with_energy_le_cw(X, 0.3)
    assert index == 5
    index = find_first_signal_with_energy_le_cw(X, 0.2)
    assert index == -1

def test_randomize_rows():
    key = random.PRNGKey(0)
    X = jnp.ones((4, 4))
    Y = randomize_rows(key, X)
    assert jnp.allclose(X, Y)

def test_randomize_cols():
    key = random.PRNGKey(0)
    X = jnp.ones((4, 4))
    Y = randomize_cols(key, X)
    assert jnp.allclose(X, Y)


def test_largest_indices():
    x = jnp.array([5, 1, 3, 4, 2])
    indices = largest_indices(x, 2)
    assert jnp.array_equal(indices, jnp.array([0, 3]))

def test_largest_indices_rw():
    x = jnp.array([[5, 1, 3, 4, 2]])
    indices = largest_indices_rw(x, 2)
    assert jnp.array_equal(indices, jnp.array([[0, 3]]))

def test_largest_indices_cw():
    x = jnp.array([[5, 1, 3, 4, 2]])
    indices = largest_indices_cw(x.T, 2)
    expected = jnp.array([[0, 3]]).T
    assert jnp.array_equal(indices, expected)

def test_take_along_rows():
    x = jnp.array([[5, 1, 3, 4, 2]])
    indices = largest_indices_rw(x, 2)
    y = take_along_rows(x, indices)
    assert jnp.array_equal(y, jnp.array([[5, 4]]))

def test_take_along_cols():
    x = jnp.array([[5, 1, 3, 4, 2]]).T
    indices = largest_indices_cw(x, 2)
    y = take_along_cols(x, indices)
    expected = jnp.array([[5, 4]]).T
    assert jnp.array_equal(y, expected)


def test_sparse_approximation():
    x = jnp.array([5, 1, 3, 4, 2])
    y = sparse_approximation(x, 2)
    expected = jnp.array([5, 0, 0, 4, 0])
    assert jnp.array_equal(y, expected)

def test_sparse_approximation_0():
    x = jnp.array([5, 1, 3, 4, 2])
    y = sparse_approximation(x, 0)
    expected = jnp.array([0, 0, 0, 0, 0])
    assert jnp.array_equal(y, expected)

def test_sparse_approximation_cw():
    x = jnp.array([[5, 1, 3, 4, 2]]).T
    y = sparse_approximation_cw(x, 2)
    expected = jnp.array([[5, 0, 0, 4, 0]]).T
    assert jnp.array_equal(y, expected)
    y = sparse_approximation_cw(x, 0)
    expected = jnp.array([[0, 0, 0, 0, 0]]).T
    assert jnp.array_equal(y, expected)


def test_sparse_approximation_rw():
    x = jnp.array([[5, 1, 3, 4, 2]])
    y = sparse_approximation_rw(x, 2)
    expected = jnp.array([[5, 0, 0, 4, 0]])
    assert jnp.array_equal(y, expected)
    y = sparse_approximation_rw(x, 0)
    expected = jnp.array([[0, 0, 0, 0, 0]])
    assert jnp.array_equal(y, expected)

def test_build_signal_from_indices_and_values():
    n = 4
    indices = jnp.array([1, 3])
    values = jnp.array([9, 15])
    x = build_signal_from_indices_and_values(n, indices, values)
    expected = jnp.array([0, 9, 0, 15])
    assert jnp.array_equal(x, expected)

def test_nonzero_values():
    x = jnp.array([0, 9, 0, 15])
    y = nonzero_values(x)
    expected = jnp.array([9, 15])
    assert jnp.array_equal(y, expected)

def test_nonzero_indices():
    x = jnp.array([0, 9, 0, 15])
    y = nonzero_indices(x)
    expected = jnp.array([1, 3])
    assert jnp.array_equal(y, expected)

def test_hard_threshold():
    x = jnp.array([5, 1, 3, 6, 2])
    I, x_I = hard_threshold(x, 2)
    assert I.size == 2
    assert x_I.size == 2
    assert jnp.array_equal(I, jnp.array([3, 0]))
    assert jnp.array_equal(x_I, jnp.array([6, 5]))

def test_hard_threshold_sorted():
    x = jnp.array([5, 1, 3, 6, 2])
    I, x_I = hard_threshold_sorted(x, 2)
    assert I.size == 2
    assert x_I.size == 2
    assert jnp.array_equal(I, jnp.array([0, 3]))
    assert jnp.array_equal(x_I, jnp.array([5, 6]))

def test_dynamic_range():
    x = jnp.array([4, -2, 3, 3, 8])
    dr = dynamic_range(x)
    assert dr >= 12 and dr <= 12.1

def test_nonzero_dynamic_range():
    x = jnp.array([4, -2, 3, 3, 8])
    dr = nonzero_dynamic_range(x)
    assert dr >= 12 and dr <= 12.1
    x = jnp.array([4, -2, 0, 0, 8])
    dr = nonzero_dynamic_range(x)
    assert dr >= 12 and dr <= 12.1


def test_SignalsComparison():
    n = 80
    s = 10
    X = jnp.ones((n,s))
    Y = 1.1 * jnp.ones((n, s))
    cmp = SignalsComparison(X, Y)
    assert len(cmp.reference_norms) == s
    assert len(cmp.estimate_norms) == s
    assert len(cmp.difference_norms) == s
    assert len(cmp.reference_energies) == s
    assert len(cmp.estimate_energies) == s
    assert len(cmp.difference_energies) == s
    assert len(cmp.error_to_signal_norms) == s
    assert len(cmp.signal_to_noise_ratios) == s
    assert cmp.cum_reference_norm
    assert cmp.cum_estimate_norm
    assert cmp.cum_difference_norm
    assert cmp.cum_error_to_signal_norm
    assert cmp.cum_signal_to_noise_ratio
    cmp.summarize()
    assert(len(snrs_cw(X, Y)) == s)
    assert(len(snrs_rw(X, Y)) == n)
    cmp = SignalsComparison(X[:,0], Y[:,0])

def test_support():
    x = jnp.concatenate((jnp.zeros(5), jnp.ones(5)))
    i = support(x)
    assert len(i) == 5

def test_hard_threshold_by():
    x = jnp.arange(10)
    y = hard_threshold_by(x, 5)
    i = support(y)
    assert len(i) == 5

def test_largest_indices_by():
    x = jnp.arange(10)
    y = largest_indices_by(x, 5)
    assert len(y) == 5

def test_normalize():
    x = jnp.arange(10) * 1.
    y = normalize(x)
    assert_almost_equal(jnp.mean(y), 0)
    assert_almost_equal(jnp.var(y), 1.)

def test_power_spectrum():
    t = jnp.linspace(0, 10, 1024)
    f = 1
    x = jnp.cos(2*jnp.pi*t)
    n = len(x)
    n2 = n // 2
    f, sxx = power_spectrum(x)
    assert len(f) == n2
    assert len(sxx) == n2
    assert jnp.all(sxx >= 0) 
