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
Dyadic index structure
-----------------------------

Let us say we are working with a signal s_J which has n=2^J samples.
1st level transform will lead to s_{J-1} with 2^{J-1} and 
d_{J - 1} with 2^{J-1} samples. 
Applying the transform again on s_{J-1}, 
we will obtain s_{J-2} with 2^{J-2} and d_{J-2} with 2^{J-2} samples.

Consider a specific case of J=4.

* s_4 has 16 samples.
* s_3 and d_3 both have 8 samples each.
* s_2 and d_2 both have 4 samples each.
* s_1 and d_1 both have 2 samples each.
* s_0 and d_0 both have 1 samples each.
* No further filtering is possible.

The final wavelet coefficients are arranged as::

    [s_0 d_0 d_1 d_2 d_3].

They occupy the indices as::

    [0, 1, 2-3, 4-7, 8-15].

s_0 is placed at the beginning.

d_j has 2^j samples and is placed between [2^j, 2^{ j + 1}-1]

* n=256 , J = 8,  we start from j=7 and go down to j=0.
* s_J = s_L + sum(L <= j < J) d_j.
* s_8 = s_0 + sum(0 <= j < 8) d_j.  [0, 1, 2-3, 4-7, 8-15, 16-31, 32-63, 64-127, 128-255]  [s_0 d_0 d_1 d_2 d_3 d_4 d_5 d_6 d_7]
* s_8 = s_4 + sum(4 <= j < 8) d_j.  [s_4(0-15), d_4(16-31), d_5(32-63), d_6(64-127), d_7(128-255)]
"""

import jax.numpy as jnp
from jax import lax
import math

def dyad(j: int):
    """Returns the indices for the entire j-th dyad of 1-d wavelet transform"""
    return jnp.arange(2**j, 2**(j+1))


def is_dyadic(j, k):
    """Verifies if the index (j, k) is a dyadic index"""
    # j , k must be integers
    c = jnp.fix(j) == j
    c = jnp.logical_and(c, jnp.fix(k) == k)
    c = jnp.logical_and(c, j >= 0)
    c = jnp.logical_and(c, k >= 0)
    c = jnp.logical_and(c, k < 2**j)
    return c


def dyad_to_index(j, k):
    """Converts wavelet indices to linear index"""
    return 2**j + k


def dyadic_length(x):
    """Returns the dydadic length of x"""
    n = x.shape[0]
    return jnp.ceil(jnp.log2(n)).astype(int)

def dyadic_length_int(x):
    """Returns the dydadic length of x"""
    n = x.shape[0]
    return math.ceil(math.log2(n))


def has_dyadic_length(x):
    """Returns if the x covers a full dyad"""
    n = x.shape[0]
    j = dyadic_length(x)
    return n == 2**j

def cut_dyadic(x):
    """Returns the part of the signal corresponding to the largest dyadic length"""
    n = x.shape[0]
    j = math.floor(math.log2(n))
    m = 2**j
    return lax.dynamic_slice(x, (0,), (m,))
