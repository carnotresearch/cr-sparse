import pytest

from jax import random
import jax.numpy as jnp

import cr.sparse.dict as crdict
import cr.sparse as crs


def test_gaussian_mtx():
    key = random.PRNGKey(0)
    M = 10
    N = 20
    A = crdict.gaussian_mtx(key, M, N, True)
    B = crdict.gaussian_mtx(key, M, N, False)


def test_rademacher_mtx():
    key = random.PRNGKey(0)
    M = 10
    N = 20
    A = crdict.rademacher_mtx(key, M, N)

def test_dirac_hadamard_basis():
    n = 16
    A = crdict.dirac_hadamard_basis(n)

def test_dirac_cosine_basis():
    n = 16
    A = crdict.dirac_cosine_basis(n)

def test_dirac_hadamard_cosine_basis():
    n = 16
    A = crdict.dirac_hadamard_cosine_basis(n)
