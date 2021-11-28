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
from jax import jit, grad
import jax.numpy as jnp



class SmoothFunction(NamedTuple):
    """Represents a smooth function
    """
    func: Callable[[jnp.ndarray], float]
    """Definition of a smooth function"""
    grad: Callable[[jnp.ndarray, float], jnp.ndarray]
    """Gradient the function"""
    grad_val: Callable[[jnp.ndarray, float], Tuple[float, jnp.ndarray]]
    "A wrapper function to evaluate the gradient vector and the function value"

def build(func):
    """Creates a wrapper for a smooth function"""
    gradient = grad(func)
    func = jit(func)
    gradient = jit(gradient)
    grad_val = build_grad_val_func(func, gradient)
    return SmoothFunction(func=func, grad=gradient, 
        grad_val=grad_val)

def build2(func, grad):
    """Creates a wrapper for a smooth function with user defined grad function
    """
    func = jit(func)
    grad = jit(grad)
    grad_val = build_grad_val_func(func, grad)
    return SmoothFunction(func=func, grad=grad, 
        grad_val=grad_val)

def build3(func, grad, grad_val):
    """Creates a wrapper for a smooth function with user defined grad and grad_val functions
    """
    func = jit(func)
    grad = jit(grad)
    grad_val = jit(grad_val)
    return SmoothFunction(func=func, grad=grad, 
        grad_val=grad_val)


def build_grad_val_func(func, grad):
    @jit
    def impl(x):
        g = grad(x)
        v = func(x)
        return g, v
    return impl
