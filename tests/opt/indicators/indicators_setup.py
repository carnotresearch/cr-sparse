import pytest

# jax imports
import jax
import jax.numpy as jnp
from jax import random, lax

# crs imports
import cr.sparse as crs
from cr.sparse.opt.indicators import *

atol = 1e-6
rtol = 1e-6


import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose, assert_,
                           assert_equal, assert_raises, assert_raises_regex,
                           assert_array_equal, assert_warns)