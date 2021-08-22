
from jax import random
import jax.numpy as jnp

import cr.sparse.data as crdata
import cr.sparse as crs

import pytest


def test_sparse_vector():
    key = random.PRNGKey(0)
    D = 10
    K = 2
    S = 1
    x, omega = crdata.sparse_normal_representations(key, D, K, S)
    x = jnp.squeeze(x)
    assert len(omega) == K
    support = crs.nonzero_indices(x)
    assert len(support) == K


def test_sparse_vectors():
    key = random.PRNGKey(0)
    D = 10
    K = 2
    S = 2
    x, omega = crdata.sparse_normal_representations(key, D, K, S)
    assert len(omega) == K
    assert x.shape == (D, S)
    support = crs.nonzero_indices(x)
    assert len(support) == K*S


def test_sparse_spikes():
    key = random.PRNGKey(0)
    N = 10
    K = 4
    x, omega = crdata.sparse_spikes(key, N, K)
    assert len(omega) == K
    assert x.shape == (N,)
    support = crs.nonzero_indices(x)
    assert len(support) == K

