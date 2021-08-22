import pytest
from functools import partial

import jax
from jax import random
import jax.numpy as jnp

import cr.sparse as crs
import cr.sparse.lop as lop
import cr.sparse.data as crdata

from cr.sparse.cvx import l1ls
from cr.sparse.cvx.adm import yall1


K = 4
M = 40
N = 200

key = random.PRNGKey(8)
keys = random.split(key, 10)
Phi = lop.gaussian_dict(keys[0], M,N)
x, omega = crdata.sparse_normal_representations(keys[1], N, K, 1)
x = jnp.squeeze(x)
y = Phi.times(x)

x_nng = jnp.abs(x)
y_nng = Phi.times(x_nng)
