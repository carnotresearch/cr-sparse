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

from jax import jit
import jax.numpy as jnp

from .util import promote_arg_dtypes

def is_scalar(x):
    return x.ndim == 0

def is_vec(x):
    return x.ndim == 1 or (x.ndim == 2 and 
        (x.shape[0] == 1 or x.shape[1] == 1))

def is_line_vec(x):
    return x.ndim == 1

def is_row_vec(x):
    return x.ndim == 2 and x.shape[0] == 1 

def is_col_vec(x):
    return x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1)

def to_row_vec(x):
    assert x.ndim == 1
    return jnp.expand_dims(x, 0)

def to_col_vec(x):
    assert x.ndim == 1
    return jnp.expand_dims(x, 1)

def vec_unit(n, i):
    """Returns a unit vector in i-th dimension"""
    return jnp.zeros(n).at[i].set(1)

vec_unit_jit = jit(vec_unit, static_argnums=(0, 1))

def vec_shift_right(x):
    """Right shift the contents of the vector"""
    return jnp.zeros_like(x).at[1:].set(x[:-1])

def vec_rotate_right(x):
    """Circular right shift the contents of the vector"""
    return jnp.roll(x, 1)


def vec_shift_left(x):
    """Left shift the contents of the vector"""
    return jnp.zeros_like(x).at[0:-1].set(x[1:])

def vec_rotate_left(x):
    """Circular left shift the contents of the vector"""
    return jnp.roll(x, -1)

def vec_shift_right_n(x, n):
    """Right shift the contents of the vector by n places"""
    return jnp.zeros_like(x).at[n:].set(x[:-n])

def vec_rotate_right_n(x, n):
    """Circular right shift the contents of the vector by n places"""
    return jnp.roll(x, n)


def vec_shift_left_n(x, n):
    """Left shift the contents of the vector by n places"""
    return jnp.zeros_like(x).at[0:-n].set(x[n:])

def vec_rotate_left_n(x, n):
    """Circular left shift the contents of the vector by n places"""
    return jnp.roll(x, -n)


def vec_repeat_at_end(x, p):
    n = x.shape[0]
    indices = jnp.arange(p) % n
    padding = x[indices]
    return jnp.concatenate((x, padding))

vec_repeat_at_end_jit = jit(vec_repeat_at_end, static_argnums=(1,))


def vec_repeat_at_start(x, p):
    n = x.shape[0]
    indices = (jnp.arange(p) + n - p) % n
    padding = x[indices]
    return jnp.concatenate((padding, x))

vec_repeat_at_start_jit = jit(vec_repeat_at_start, static_argnums=(1,))


def vec_centered(x, length):
    cur_len = len(x)
    length = min(cur_len, length) 
    start = (len(x) - length) // 2
    end = start + length
    return x[start:end]

vec_centered_jit = jit(vec_centered, static_argnums=(1,))
