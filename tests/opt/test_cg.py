import pytest

# jax imports
import jax
from jax import jit, random
import jax.numpy as jnp

# crs imports
from cr.sparse.opt import cg

def test_cg():
    A = jnp.array([[3., 2], [2, 6]])
    b = jnp.array([2., -8])
    sol = cg.solve_jit(A, b)
    r = A @ sol.x - b
    assert jnp.isclose(0, r @ r)
