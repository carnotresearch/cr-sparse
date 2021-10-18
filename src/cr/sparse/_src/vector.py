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
Utility functions for working with vectors
"""

from jax import jit, lax
import jax.numpy as jnp
from jax.scipy import signal


from .util import promote_arg_dtypes

def is_scalar(x):
    """Returns if x is a scalar 

    Args:
        x (jax.numpy.ndarray): A JAX array.

    Returns:
        True if x is a scalar quantity (i.e. ndim==0).
    """
    return x.ndim == 0

def is_vec(x):
    """Returns if x is a line vector or row vector or column vector

    Args:
        x (jax.numpy.ndarray): A JAX array.

    Returns:
        True if x is a line vector or a row vector or a column vector.
    """
    return x.ndim == 1 or (x.ndim == 2 and 
        (x.shape[0] == 1 or x.shape[1] == 1))

def is_line_vec(x):
    """Returns if x is a line vector

    Args:
        x (jax.numpy.ndarray): A JAX array.

    Returns:
        True if x is a line vector.
    """
    return x.ndim == 1

def is_row_vec(x):
    """Returns if x is a row vector

    Args:
        x (jax.numpy.ndarray): A JAX array.

    Returns:
        True if x is a row vector.
    """
    return x.ndim == 2 and x.shape[0] == 1 

def is_col_vec(x):
    """Returns if x is a column vector

    Args:
        x (jax.numpy.ndarray): A JAX array.

    Returns:
        True if x is a column vector.
    """
    return x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1)

def to_row_vec(x):
    """Converts a line vector to a row vector

    Args:
        x (jax.numpy.ndarray): A line vector (ndim == 1).

    Returns:
        jax.numpy.ndarray: A row vector.
    """
    assert x.ndim == 1
    return jnp.expand_dims(x, 0)

def to_col_vec(x):
    """Converts a line vector to a column vector

    Args:
        x (jax.numpy.ndarray): A line vector (ndim == 1).

    Returns:
        jax.numpy.ndarray: A column vector.
    """
    assert x.ndim == 1
    return jnp.expand_dims(x, 1)

def vec_unit(n, i):
    """Returns a unit vector in i-th dimension for the standard coordinate system

    Args:
        n (int): Length of the vector.
        i (int): Index/dimension of the unit vector.

    Returns:
        jax.numpy.ndarray: A line vector of length n with all zeros except a one at position i. 
    """
    return jnp.zeros(n).at[i].set(1)

vec_unit_jit = jit(vec_unit, static_argnums=(0, 1))

def vec_shift_right(x):
    """Right shift the contents of the vector

    Args:
        x (jax.numpy.ndarray): A line vector.

    Returns:
        jax.numpy.ndarray: Right shifted x. 
    """
    return jnp.zeros_like(x).at[1:].set(x[:-1])

def vec_rotate_right(x):
    """Circular right shift the contents of the vector

    Args:
        x (jax.numpy.ndarray): A line vector.

    Returns:
        jax.numpy.ndarray: Right rotated x. 
    """
    return jnp.roll(x, 1)


def vec_shift_left(x):
    """Left shift the contents of the vector

    Args:
        x (jax.numpy.ndarray): A line vector.

    Returns:
        jax.numpy.ndarray: Left shifted x. 
    """
    return jnp.zeros_like(x).at[0:-1].set(x[1:])

def vec_rotate_left(x):
    """Circular left shift the contents of the vector

    Args:
        x (jax.numpy.ndarray): A line vector.

    Returns:
        jax.numpy.ndarray: Left rotated x. 
    """
    return jnp.roll(x, -1)

def vec_shift_right_n(x, n):
    """Right shift the contents of the vector by n places

    Args:
        x (jax.numpy.ndarray): A line vector.
        n (int): Number of positions to shift.

    Returns:
        jax.numpy.ndarray: Right shifted x by n places. 
    """
    return jnp.zeros_like(x).at[n:].set(x[:-n])

def vec_rotate_right_n(x, n):
    """Circular right shift the contents of the vector by n places

    Args:
        x (jax.numpy.ndarray): A line vector.
        n (int): Number of positions to shift.

    Returns:
        jax.numpy.ndarray: Right roted x by n places. 
    """
    return jnp.roll(x, n)


def vec_shift_left_n(x, n):
    """Left shift the contents of the vector by n places

    Args:
        x (jax.numpy.ndarray): A line vector.
        n (int): Number of positions to shift.

    Returns:
        jax.numpy.ndarray: Left shifted x by n places. 
    """
    return jnp.zeros_like(x).at[0:-n].set(x[n:])

def vec_rotate_left_n(x, n):
    """Circular left shift the contents of the vector by n places

    Args:
        x (jax.numpy.ndarray): A line vector.
        n (int): Number of positions to shift.

    Returns:
        jax.numpy.ndarray: Left rotated x by n places. 
    """
    return jnp.roll(x, -n)


def vec_repeat_at_end(x, p):
    """Extends a vector by repeating it at the end (periodic extension)

    Args:
        x (jax.numpy.ndarray): A line vector.
        p (int): Number of samples by which x will be extended.

    Returns:
        jax.numpy.ndarray: x extended periodically at the end. 
    """
    n = x.shape[0]
    indices = jnp.arange(p) % n
    padding = x[indices]
    return jnp.concatenate((x, padding))

vec_repeat_at_end_jit = jit(vec_repeat_at_end, static_argnums=(1,))


def vec_repeat_at_start(x, p):
    """Extends a vector by repeating it at the start (periodic extension)

    Args:
        x (jax.numpy.ndarray): A line vector.
        p (int): Number of samples by which x will be extended.

    Returns:
        jax.numpy.ndarray: x extended periodically at the start. 
    """
    n = x.shape[0]
    indices = (jnp.arange(p) + n - p) % n
    padding = x[indices]
    return jnp.concatenate((padding, x))

vec_repeat_at_start_jit = jit(vec_repeat_at_start, static_argnums=(1,))


def vec_centered(x, length):
    """Returns the central part of a vector of a specified length

    Args:
        x (jax.numpy.ndarray): A line vector.
        length (int): Length of the central part of x which will be retained.

    Returns:
        jax.numpy.ndarray: central part of x of the specified length. 
    """
    cur_len = len(x)
    length = min(cur_len, length) 
    start = (len(x) - length) // 2
    end = start + length
    return x[start:end]

vec_centered_jit = jit(vec_centered, static_argnums=(1,))


def vec_convolve(x, h):
    """1D full convolution based on a hack suggested by Jake Vanderplas

    See https://github.com/google/jax/discussions/7961 for details
    """
    return signal.convolve(x[None], h[None])[0]

vec_convolve_jit = jit(vec_convolve)


def vec_safe_divide_by_scalar(x, alpha):
    return lax.cond(alpha == 0, lambda x : x, lambda x: x / alpha, x)

vec_safe_divide_by_scalar_jit = jit(vec_safe_divide_by_scalar)