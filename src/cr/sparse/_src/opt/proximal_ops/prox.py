# Copyright 2021 CR-Suite Development Team
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
    r"""Represents a function which is prox capable

    The *proximal operator* for a function :math:`f` is defined as

    .. math::

        p_f(x, t) = \text{arg} \min_{z \in \RR^n} f(x) + \frac{1}{2t} \| z - x \|_2^2

    Let `op` be a variable of type ProxCapable 
    which represents some prox-capable function :math:`f`. Then:

    * `op.func(x)` returns the function value :math:`f(x)`.
    * `op.prox_op(x)` returns the proximal vector for a step size: :math:`z = p_f(x, t)`.
    * `op.prox_vec_val(x)` returns the pair :math:`z,v = p_f(x, t), f(z)`.
    """
    func: Callable[[jnp.ndarray], float]
    """Definition of a prox capable function"""
    prox_op: Callable[[jnp.ndarray, float], jnp.ndarray]
    """Proximal operator for the function"""
    prox_vec_val: Callable[[jnp.ndarray, float], Tuple[float, jnp.ndarray]]
    "A wrapper function to evaluate the proximal vector and the function value at the vector"


def build(func, prox_op):
    r"""Creates a wrapper for a prox capable function

    Args:
        func: Definition of a a function :math:`f(x)`
        prox_op: Definition of its proximal operator :math:`p_f(x, t)`

    Returns:
       ProxCapable: A prox-capable function 
    """
    func = jit(func)
    prox_op = jit(prox_op)
    prox_vec_val = build_prox_value_vec_func(func, prox_op)
    return ProxCapable(func=func, prox_op=prox_op, 
        prox_vec_val=prox_vec_val)

def build3(func, prox_op, prox_vec_val):
    r"""Creates a wrapper for a prox capable function

    Args:
        func: Definition of a a function :math:`f(x)`
        prox_op: Definition of its proximal operator :math:`p_f(x, t)`
        prox_vec_val: Combined function for generating both proximal point and value

    Returns:
       ProxCapable: A prox-capable function 
    """
    return ProxCapable(func=func, prox_op=prox_op, 
        prox_vec_val=prox_vec_val)

def build_from_ind_proj(indicator, projector):
    """Builds a prox capable function wrapper for a convex set indicator function

    Args:
        indicator: Definition of the indicator function for the convex set
        projector: Definition of the projector function for the convex set

    Returns:
       ProxCapable: A prox-capable function 
    """
    indicator = jit(indicator)

    @jit
    def prox_op(x, t):
        return projector(x)

    @jit
    def prox_vec_val(x, t):
        # first project to the convex set
        z = projector(x)
        # the value of indicator function inside the convex set is 0.
        return z, 0.

    return ProxCapable(func=indicator, prox_op=prox_op, 
        prox_vec_val=prox_vec_val)


def build_prox_value_vec_func(func, prox_op):
    r"""Creates function which computes the proximal vector and then function value at it 

    Args:
        func: Definition of a a function :math:`f(x)`
        prox_op: Definition of its proximal operator :math:`p_f(x, t)`

    Returns:
        A function which can compute the pair :math:`z,v = p_f(x,t), f(z)`
    """
    @jit
    def impl(x, t):
        x = prox_op(x, t)
        v = func(x)
        return x, v
    return impl
