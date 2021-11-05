# Copyright 2021 CR.Sparse Development Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utility functions for ND arrays
"""

from jax import jit
import jax.numpy as jnp

from .util import promote_arg_dtypes


def arr_largest_index(x):
    """Returns the unraveled index of the largest entry (by magnitude) in an n-d array

    Args:
        x (jax.numpy.ndarray): An nd-array

    Returns:
        tuple: n-dim index of the largest entry in x 
    """
    x = jnp.asarray(x)
    return jnp.unravel_index(jnp.argmax(jnp.abs(x)), x.shape)


def arr_l2norm(x):
    """Returns the l2-norm of an array by flattening it
    """
    return jnp.sqrt(jnp.abs(jnp.vdot(x, x)))

def arr_l2norm_sqr(x):
    """Returns the squared l2-norm of an array by flattening it
    """
    return jnp.vdot(x, x)

def arr_vdot(x, y):
    """Returns the inner product of two arrays  by flattening it 
    """
    return jnp.vdot(x, y)

@jit
def arr_rdot(x, y):
    """Returns the inner product Re(x^H, y) on two arrays by flattening them
    """
    x = jnp.ravel(x)
    y = jnp.ravel(y)
    if jnp.isrealobj(x) and jnp.isrealobj(y):
        # we can fall back to real inner product
        return jnp.sum(x * y)
    if jnp.isrealobj(x) or jnp.isrealobj(y):
        # 
        x = jnp.real(x)
        y = jnp.real(y)
        return jnp.sum(x * y)
    # both x and y are complex
    # compute x^H
    x = jnp.conjugate(x)    
    # compute x^H y
    prod = jnp.sum(x * y)
    # take the real part
    return jnp.real(prod)


@jit
def arr2vec(x):
    """Converts an nd array to a vector
    """
    return jnp.ravel(x)