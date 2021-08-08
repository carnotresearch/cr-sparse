import pytest

import numpy as np
from numpy.testing import assert_allclose, assert_, assert_raises

# from jax.config import config
# config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
from jax import random


from cr.sparse import *
import cr.sparse.wt as wt

key = random.PRNGKey(0)
keys = random.split(key, 16)
