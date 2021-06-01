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

import math
import jax.numpy as jnp
from sympy.ntheory import factorint

def is_integer(x):
    return jnp.mod(x, 1) == 0

def is_positive_integer(x):
    return jnp.logical_and(x > 0, jnp.mod(x, 1) == 0)

def is_negative_integer(x):
    return jnp.logical_and(x < 0, jnp.mod(x, 1) == 0)


def is_odd(x):
    return jnp.mod(x, 2) == 1

def is_even(x):
    return jnp.mod(x, 2) == 0

def is_odd_natural(x):
    return jnp.logical_and(x > 0, jnp.mod(x, 2) == 1)


def is_even_natural(x):
    return jnp.logical_and(x > 0, jnp.mod(x, 2) == 0)

def is_power_of_2(x):
    return jnp.logical_not(jnp.bitwise_and(x, x - 1))

def is_perfect_square(x):
    return is_integer(jnp.sqrt(x))


def integer_factors_close_to_sqr_root(n):
    assert isinstance(n, int)
    a_max = math.floor(math.sqrt(n))
    if n % a_max == 0:
        a = a_max
        b = n // a
        return a,b
    # get the prime factors
    factors_map = factorint(n)
    factors = factors_map.keys()
    candidates = {1}
    #print(a_max)
    for key in factors_map:
        for count in range(factors_map[key]):
            new_candidates = {key*c for c in candidates}
            candidates = candidates.union(new_candidates)
            # filter out larger candidates
            candidates = {c for c in candidates if c <= a_max}
            #print(candidates)
    # a is the last candidate
    candidates = list(candidates)
    candidates.sort()
    a = candidates[-1]
    b = n // a
    return a, b