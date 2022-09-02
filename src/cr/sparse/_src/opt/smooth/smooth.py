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
from jax import jit, grad
import jax.numpy as jnp

import cr.nimble as cnb


class SmoothFunction(NamedTuple):
    r"""Represents a smooth function

    Let `op` be a variable of type `SmoothFunction`
    which represents some smooth function :math:`f`. Then:

    * `op.func(x)` returns the function value :math:`f(x)`.
    * `op.grad(x)` returns the gradient of function :math:`g(x) = \nabla f(x)`.
    * `op.grad_val(x)` returns the pair :math:`(g(x), f(x))`.
    """
    func: Callable[[jnp.ndarray], float]
    """Definition of a smooth function"""
    grad: Callable[[jnp.ndarray, float], jnp.ndarray]
    """Definition of a gradient the function"""
    grad_val: Callable[[jnp.ndarray, float], Tuple[float, jnp.ndarray]]
    "A wrapper to evaluate the gradient vector and the function value together"

def build(func):
    r"""Creates a smooth function based on function definition :math:`f(x)`

    Args:
        func: Definition of the smooth function :math:`f : \RR^n \to \RR`

    Returns:
        SmoothFunction: A smooth function wrapper
    """
    gradient = grad(func)
    func = jit(func)
    gradient = jit(gradient)
    grad_val = build_grad_val_func(func, gradient)
    return SmoothFunction(func=func, grad=gradient, 
        grad_val=grad_val)

def build2(func, grad):
    r"""Creates a smooth function with user defined :math:`f(x)` and gradient :math:`g(x)`

    Args:
        func: Definition of the smooth function :math:`f : \RR^n \to \RR`
        grad: Definition of the gradient :math:`g = \nabla f : \RR^n \to \RR^n`

    Returns:
        SmoothFunction: A smooth function wrapper
    """
    func = jit(func)
    grad = jit(grad)
    grad_val = build_grad_val_func(func, grad)
    return SmoothFunction(func=func, grad=grad, 
        grad_val=grad_val)

def build3(func, grad, grad_val):
    r"""Creates a a smooth function with user defined grad and grad_val functions

    Args:
        func: Definition of the smooth function :math:`f : \RR^n \to \RR`
        grad: Definition of the gradient :math:`g = \nabla f : \RR^n \to \RR^n`
        grad_val: Definition of a combined function which computes the pair :math:`(g(x), f(x))`

    Returns:
        SmoothFunction: A smooth function wrapper
    """
    func = jit(func)
    grad = jit(grad)
    grad_val = jit(grad_val)
    return SmoothFunction(func=func, grad=grad, 
        grad_val=grad_val)


def build_grad_val_func(func, grad):
    r"""Constructs a `grad_val` function from the definitions of function :math:`f(x)` and gradient :math:`g(x)`

    Args:
        func: Definition of the smooth function :math:`f : \RR^n \to \RR`
        grad: Definition of the gradient :math:`g = \nabla f : \RR^n \to \RR^n`

    Returns:
        A function which computes the pair :math:`(g(x), f(x))` for input :math:`x`
    """
    @jit
    def impl(x):
        g = grad(x)
        v = func(x)
        return g, v
    return impl


def smooth_func_translate(smooth_func, b):
    r"""Returns a smooth function :math:`g` for a smooth function :math:`f` s.t. :math:`g(x) = f(x + b)`

    Args:
        smooth_func (SmoothFunction): Wrapper for smooth function :math:`f : \RR^n \to \RR`
        b (jax.numpy.ndarray): The offset/translation vector :math:`b \in \RR^n`

    Returns:
        SmoothFunction: A smooth function wrapper for the function :math:`g` such that
        :math:`g(x) = f(x+b)`
    """
    b = jnp.asarray(b)
    b = cnb.promote_arg_dtypes(b)

    @jit
    def func(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        return smooth_func.func(x + b)
    @jit
    def grad(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        return smooth_func.grad(x + b)
    @jit
    def grad_val(x):
        x = jnp.asarray(x)
        x = cnb.promote_arg_dtypes(x)
        return smooth_func.grad_val(x + b)
    return SmoothFunction(func=func, grad=grad, 
        grad_val=grad_val)
