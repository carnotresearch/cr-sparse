import pytest

# jax imports
import jax
import jax.numpy as jnp

# crs imports
import cr.sparse as crs
from cr.sparse.dsp import *

atol = 1e-6
rtol = 1e-6

def test_dct1():
    for n in [4,8,16]:
        for i in range(n):
            print(n, i)
            y = crs.vec_unit(n, i)
            a = dct(y)
            x = idct(a)
            assert jnp.allclose(x, y, rtol=rtol, atol=atol)

def test_orthonormal_dct1():
    for n in [4,8,16]:
        for i in range(n):
            print(n, i)
            y = crs.vec_unit(n, i)
            a = orthonormal_dct(y)
            x = orthonormal_idct(a)
            assert jnp.allclose(x, y, rtol=rtol, atol=atol)