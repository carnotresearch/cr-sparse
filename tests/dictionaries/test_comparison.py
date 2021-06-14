import pytest

from jax import random
import jax.numpy as jnp

import cr.sparse.dict as crdict
import cr.sparse as crs

def test_comparison():
    key = random.PRNGKey(0)
    N = 16
    A = crdict.random_onb(key, N)
    ratio = crdict.matching_atoms_ratio(A,A)
    assert jnp.equal(ratio, 1.0)

