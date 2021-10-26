import pytest


import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose, assert_,
                           assert_equal, assert_raises, assert_raises_regex,
                           assert_array_equal, assert_warns)


import cr.sparse as crs
import jax.numpy as jnp

def test_dist_to_gaussian_sim():
    n = 10
    I = jnp.eye(n)
    distances = jnp.ones((n, n)) - I
    sim = crs.dist_to_gaussian_sim(distances, sigma=1.)
    expected =  0.606531 * jnp.ones((n, n))
    expected = crs.set_diagonal(expected,  1)
    assert_allclose(sim, expected, atol=1e-5)

def test_sqr_dist_to_gaussian_sim():
    n = 10
    I = jnp.eye(n)
    distances = jnp.ones((n, n)) - I
    sqr_dist = distances ** 2
    sim = crs.sqr_dist_to_gaussian_sim(sqr_dist, sigma=1.)
    expected =  0.606531 * jnp.ones((n, n))
    expected = crs.set_diagonal(expected,  1)
    assert_allclose(sim, expected, atol=1e-5)

def test_eps_neighborhood_sim():
    n = 10
    I = jnp.eye(n)
    distances = jnp.ones((n, n)) - I
    sim = crs.eps_neighborhood_sim(distances, 0.5)
    assert_allclose(sim, I)
