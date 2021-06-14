import pytest

# jax imports
import jax
import jax.numpy as jnp

# crs imports
from cr.sparse.la import *

def test_point():
    x = point2d(1, 2)

def test_vec():
    x = vec2d(1, 2)

def test_rotate2d_cw():
    theta = jnp.pi
    R = rotate2d_cw(theta)

def test_rotate2d_ccw():
    theta = jnp.pi
    R = rotate2d_ccw(theta)

def test_reflect2d():
    theta = jnp.pi
    R = reflect2d(theta)
