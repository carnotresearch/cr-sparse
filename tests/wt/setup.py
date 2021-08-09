import pytest

import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose, assert_,
                           assert_equal, assert_raises, assert_raises_regex,
                           assert_array_equal, assert_warns)

# from jax.config import config
# config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
from jax import random, lax


from cr.sparse import *
import cr.sparse.wt as wt

key = random.PRNGKey(0)
keys = random.split(key, 16)

rtol = 1e-8 if jax.config.jax_enable_x64 else 1e-6
atol = 1e-7 if jax.config.jax_enable_x64 else 1e-5
decimal_cmp = 7 if jax.config.jax_enable_x64 else 6

float_type = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
complex_type = jnp.complex128 if jax.config.jax_enable_x64 else jnp.complex64

# There are some changes in the way input and output types are related.
# all our output is float32 or float64 depending on jax_enable_x64 configuration
dtypes_in = [jnp.int8, jnp.float16, 
    jnp.float32, jnp.float64, 
    jnp.complex64, jnp.complex128]

dtypes_out = [float_type, float_type, 
    float_type, float_type, 
    complex_type, complex_type]


discrete_wavelets = wt.wavelist(kind='discrete')