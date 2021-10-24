import pytest

# jax imports
import jax
import jax.numpy as jnp
from jax import random, lax

# crs imports
import cr.sparse as crs
from cr.sparse import lop

rtol = 1e-8 if jax.config.jax_enable_x64 else 1e-6
atol = 1e-7 if jax.config.jax_enable_x64 else 1e-5


import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose, assert_,
                           assert_equal, assert_raises, assert_raises_regex,
                           assert_array_equal, assert_warns)


key = random.PRNGKey(0)
keys = random.split(key, 16)
