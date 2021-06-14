import pytest

# jax imports
import jax
import jax.numpy as jnp

# crs imports
from cr.sparse.la import rq


def test_1():
    A = jnp.eye(3)
    R, Q = rq.factor_mgs(A)

def test_2():
    with pytest.raises(Exception):
        R, Q = rq.factor_mgs(jnp.zeros((5, 3)))

def test_3():
    A = jnp.eye(3)
    n, m = A.shape
    Q = jnp.empty([n, m])
    R = jnp.zeros([n, n])
    R, Q = rq.update(R, Q, A[0], 0)
    R, Q = rq.update(R, Q, A[1], 1)


def test_4():
    A = jnp.eye(3)
    n, m = A.shape
    Q = jnp.empty([n, m])
    R = jnp.zeros([n, n])
    x = jnp.zeros(n)
    rq.solve(R, Q, x)
