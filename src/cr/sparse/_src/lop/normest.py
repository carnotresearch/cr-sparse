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

"""Power iteration method for linear operators
"""

from jax import jit, lax, random
import jax.numpy as jnp

from typing import NamedTuple

import cr.nimble as cnb

class NormEstState(NamedTuple):
    """State for the norm estimation algorithm
    """
    # (unnormalized) eigen vector guess
    v: jnp.ndarray
    # eigen value estimate
    old_estimate: float
    # new eigen value estimate
    new_estimate: float
    # number of iterations
    iterations: int

class NormEstSolution(NamedTuple):
    """Solution of the eigen vector estimate
    """
    v: jnp.ndarray
    """The estimated eigen vector"""
    s : float
    """The estimated eigen value"""
    iterations: int
    """Number of iterations to converge"""



def normest(
    operator,
    max_iters=100,
    error_tolerance=1e-6):
    """Estimates the norm of a linear operator by power method

    Args:
        operator (cr.sparse.lop.Operator): A linear operator :math:`A`
        max_iters (int): Maximum number of iterations
        error_tolerance (float): Tolerance for relative change in largest eigen value 

    Returns:
        (float): An estimate of the norm
    """
    shape = operator.input_shape
    # initial eigen vector
    b = random.normal(cnb.KEYS[0], shape)
    if not operator.real:
        bi = random.normal(cnb.KEYS[1], shape)
        b = b + bi * 1j
    # normalize it
    b = b / cnb.arr_l2norm(b)
    def init():
        return NormEstState(v=b, old_estimate=-1e20, new_estimate=1e20, iterations=0)

    def cond(state):
        # check if the gap between new and old estimate is still high
        change = state.new_estimate - state.old_estimate
        relchange =  jnp.abs(change / state.old_estimate)
        not_converged = jnp.greater(relchange, error_tolerance)
        # return true if the the algorithm hasn't converged and there are more iterations to go
        return jnp.logical_and(state.iterations < max_iters, not_converged)

    def body(state):
        """One step of power iteration."""
        v = state.v
        # normalize
        v_norm = cnb.arr_l2norm(v)
        v = v / v_norm
        # compute the next vector
        v_new = operator.times(v)
        v_new = operator.trans(v_new)
        # estimate the eigen value
        new_estimate = jnp.vdot(v, v_new)
        # largest singular value is non-negative
        new_estimate = jnp.abs(new_estimate)
        return NormEstState(v=v_new, old_estimate=state.new_estimate, 
            new_estimate=new_estimate, iterations=state.iterations+1)

    state = lax.while_loop(cond, body, init())
    # We have converged
    return jnp.sqrt(state.new_estimate)

normest_jit = jit(normest, static_argnums=(0,))

