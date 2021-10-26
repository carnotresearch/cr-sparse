import pytest

import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose, assert_,
                           assert_equal, assert_raises, assert_raises_regex,
                           assert_array_equal, assert_warns)

import jax
import jax.numpy as jnp
from jax import random, lax, vmap


import cr.sparse as crs
import cr.sparse.la as la
import cr.sparse.data as crdata
import cr.sparse.cluster as cluster
import cr.sparse.cluster.ssc as ssc

rtol = 1e-8 if jax.config.jax_enable_x64 else 1e-6
atol = 1e-7 if jax.config.jax_enable_x64 else 1e-5
