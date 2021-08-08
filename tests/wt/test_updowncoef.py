import pytest

import numpy as np
from numpy.testing import assert_allclose, assert_, assert_raises

import jax.numpy as jnp
from jax import random


from cr.sparse import *
import cr.sparse.wt as wt

key = random.PRNGKey(0)
keys = random.split(key, 16)


def test_upcoef_and_downcoef_1d_only():
    # upcoef and downcoef raise a ValueError if data.ndim > 1d
    for ndim in [2, 3]:
        data = jnp.ones((8, )*ndim)
        assert_raises(ValueError, wt.downcoef, 'a', data, 'haar')
        assert_raises(ValueError, wt.upcoef, 'a', data, 'haar')
