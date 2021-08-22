import pytest
import math
# jax imports
import jax
import jax.numpy as jnp

# crs imports
import cr.sparse as crs
from cr.sparse.dsp import *

atol = 1e-6
rtol = 1e-6

def test_fwht1():
    for n in [4,8,16]:
        fact = 1/math.sqrt(n)
        for i in range(n):
            print(n, i)
            y = crs.vec_unit(n, i)
            a = fact * fwht(y) 
            x = fact * fwht(a)
            assert jnp.allclose(x, y, rtol=rtol, atol=atol)