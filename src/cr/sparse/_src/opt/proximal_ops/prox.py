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


from functools import reduce
from typing import NamedTuple, Callable, Tuple
import jax
from jax import jit
import jax.numpy as jnp



class ProxCapable(NamedTuple):
    """Represents a function which is prox capable
    """
    func: Callable[[jnp.ndarray], float]
    """Definition of a prox capable function"""
    prox_op: Callable[[jnp.ndarray, float], jnp.ndarray]
    """Proximal operator for the function"""
    prox_vec_val: Callable[[jnp.ndarray, float], Tuple[float, jnp.ndarray]]
    "A wrapper function to evaluate the proximal vector and the function value at the vector"


def build(func, prox_op):
    """Creates a wrapper for a prox capable function 
    """
    func = jit(func)
    prox_op = jit(prox_op)
    prox_vec_val = build_prox_value_vec_func(func, prox_op)
    return ProxCapable(func=func, prox_op=prox_op, 
        prox_vec_val=prox_vec_val)


def build_prox_value_vec_func(func, prox_op):
    """Creates function which computes the proximal vector and then function value at it 
    """
    @jit
    def impl(x, t):
        x = prox_op(x, t)
        v = func(x)
        return x, v
    return impl
