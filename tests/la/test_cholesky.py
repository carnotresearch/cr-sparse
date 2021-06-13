import pytest

# jax imports
import jax
from jax import jit, random
import jax.numpy as jnp

# crs imports
import cr.sparse.la as crla

cholesky_build_factor = jit(crla.cholesky_build_factor)

def test_cholesky_update():
    key = random.PRNGKey(0)
    A = random.normal(key, (4,4))
    L = cholesky_build_factor(A)
    G1 = A.T @ A
    G2 = L @ L.T
    assert jnp.allclose(G1, G2)