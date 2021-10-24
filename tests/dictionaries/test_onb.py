import pytest

from jax import random
import jax.numpy as jnp

import cr.sparse.dict as crdict
import cr.sparse as crs

def test_random_onb():
    key = random.PRNGKey(0)
    N = 16
    A = crdict.random_onb(key, N)
    assert crs.is_matrix(A)
    assert crs.is_square(A)
    assert A.shape == (N, N)
    assert crs.has_orthogonal_columns(A)
    assert crs.has_orthogonal_rows(A)


def test_hadamard_basis():
    N = 16
    A = crdict.hadamard_basis(N)
    assert crs.is_matrix(A)
    assert crs.is_square(A)
    assert A.shape == (N, N)
    assert crs.has_orthogonal_columns(A)
    assert crs.has_orthogonal_rows(A)
    G = crdict.gram(A)
    assert crs.is_symmetric(G)
    if crs.is_cpu():
        assert crs.is_positive_definite(G)
    F = crdict.frame(A)
    assert crs.is_symmetric(F)
    if crs.is_cpu():
        assert crs.is_positive_definite(F)

def test_cosine_basis():
    N = 16
    A = crdict.cosine_basis(N)
    assert crs.is_matrix(A)
    assert crs.is_square(A)
    assert A.shape == (N, N)
    assert crs.has_orthogonal_columns(A)
    assert crs.has_orthogonal_rows(A)
    G = crs.transpose(A) @ A
    assert crs.is_symmetric(G)
    if crs.is_cpu():
        assert crs.is_positive_definite(G)
    mu = crdict.coherence(A)
    assert mu < 1e-3
    bounds = crdict.frame_bounds(A)
    assert len(bounds) == 2
    assert jnp.isclose(bounds[0], 1, atol=1e-4)
    assert jnp.isclose(bounds[1], 1, atol=1e-4)
    assert jnp.isclose(crdict.upper_frame_bound(A), 1, atol=1e-4)
    assert jnp.isclose(crdict.lower_frame_bound(A), 1, atol=1e-4)
    mu = crdict.mutual_coherence(A, A)
    assert jnp.isclose(mu, 1, atol=1e-4)
    b = crdict.babel(A)
    assert len(b) == N-1 


def test_fourier_basis():
    N = 16
    A = crdict.fourier_basis(N)
    assert crs.is_matrix(A)
    assert crs.is_square(A)
    assert A.shape == (N, N)
    assert crs.has_unitary_columns(A)
    assert crs.has_unitary_rows(A)
    G = crs.hermitian(A) @ A
    assert crs.is_hermitian(G)
    G = crdict.gram(A)
    assert crs.is_hermitian(G)
    F = crdict.frame(A)
    assert crs.is_hermitian(F)
