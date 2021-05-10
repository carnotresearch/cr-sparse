import math
import jax.numpy as jnp
from jax import random
from jax import jit
from cr.sparse.norm import normalize_l2_rw

def gaussian_mtx(key, N, D, normalize_atoms=True):
    shape = (D, N)
    dict = random.normal(key, shape)
    if normalize_atoms:
        dict = normalize_l2_rw(dict)
    else:
        sigma = math.sqrt(N)
        dict = dict / sigma
    return dict
