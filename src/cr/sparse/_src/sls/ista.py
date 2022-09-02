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

"""Iterative shrinkage and thresholding algorithm
"""

from jax import jit, lax
import jax.numpy as jnp

from typing import NamedTuple

from cr.nimble import arr_l2norm, arr_l2norm_sqr, arr_vdot
from .defs import identity_op, default_threshold

class ISTAState(NamedTuple):
    """ISTA algorithm state
    """
    x : jnp.ndarray
    """Current solution estimate"""
    r : jnp.ndarray
    """Current residual"""
    r_norm_sqr: jnp.ndarray
    """Square of residual norm"""
    x_change_norm: jnp.ndarray
    """Change in the norm of x """
    iterations: int
    """Number of iterations to converge"""

def ista(
    operator,
    b,
    x0,
    step_size,
    threshold_func=default_threshold,
    basis=identity_op,
    res_norm_rtol=1e-3,
    x_norm_change_tol=1e-10,
    max_iters=1000,
    ):
    r"""Solves the problem :math:`\widehat{x} = \text{arg} \min_{x} \frac{1}{2}\| b - A x \|_2^2 + \lambda \mathbf{R}(x)` 
    via iterative shrinkage and thresholding.

    It supports the more general problem where :math:`x` is sparse in a basis :math:`B` 

    .. math::

        \widehat{x} = \text{arg} \min_{x} \frac{1}{2}\| b - A x \|_2^2 + \lambda \mathbf{R} (B^H x) 

    Args:
        operator (cr.sparse.lop.Operator): A linear operator :math:`A`
        b (jax.numpy.ndarray): Data vector
        x0 (jax.numpy.ndarray): An initial estimate :math:`x_0` of the model vector :math:`x`
        step_size (float): Step size for ISTA iteration
        threshold_func (Function): A user defined thresholding function. See :ref:`sls:thresholding` for details.
        basis (cr.sparse.lop.Operator): A sparsifying basis :math:`B` for :math:`x`
        res_norm_rtol (float): Relative tolerance for norm of residual :math:`r = b - A x` relative to norm of :math:`b`
        x_norm_change_tol (float): Tolerance for change in :math:`x` in each iteration
        max_iters (int): Maximum number of iterations 

    Returns:
        ISTAState: a named tuple containing the solution :math:`x`, the residual :math:`r`
        and other details.

    See :ref:`sls:ista` for an introduction to IST algorithm.

    Both ``x0`` and ``b`` can be multi-dimensional arrays. 
    """
    m, n = operator.shape

    b_norm_sqr = arr_l2norm_sqr(b)
    r_norm_sqr_threshold = b_norm_sqr * (res_norm_rtol ** 2)

    def init():
        # compute the initial residual
        r = b - operator.times(x0)
        # compute the norm of the initial residual
        r_norm_sqr = arr_l2norm_sqr(r)
        return ISTAState(x=x0, r=r, r_norm_sqr=r_norm_sqr,
            x_change_norm=1e10, 
            iterations=0)

    def body(state):
        # compute the gradient step
        grad = step_size * operator.trans(state.r)
        # update the solution
        x = state.x + grad
        # compute the representation of x in the sparsifying basis
        alpha = basis.trans(x)
        # apply the thresholding function on the sparse representation
        alpha = threshold_func(state.iterations, alpha)
        # convert back to data space
        x = basis.times(alpha)
        # update the residual
        r = b - operator.times(x)
        # compute the norm of the current residual
        r_norm_sqr = arr_l2norm_sqr(r)
        x_change_norm = arr_l2norm(x - state.x)
        return ISTAState(x=x, r=r, r_norm_sqr=r_norm_sqr, 
            x_change_norm=x_change_norm, iterations=state.iterations+1)

    def cond(state):
        not_converged = jnp.greater(state.r_norm_sqr, r_norm_sqr_threshold)
        not_converged = jnp.logical_and(state.x_change_norm > x_norm_change_tol, not_converged)
        # return true if the the algorithm hasn't converged and there are more iterations to go
        return jnp.logical_and(state.iterations < max_iters, not_converged)

    state = lax.while_loop(cond, body, init())
    return state


ista_jit = jit(ista, static_argnums=(0, 3, 4, 5, 6, 7, 8))
