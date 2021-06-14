import pytest

import jax
from jax import random, jit
import jax.numpy as jnp

from cr.sparse.pursuit import phasetransition

def test_1():
    conf = phasetransition.configuration(64)
    
def test_2():
    conf = phasetransition.configuration(512)
