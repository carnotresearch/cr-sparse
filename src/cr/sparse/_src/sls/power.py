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

from jax import jit, lax
import jax.numpy as jnp

from typing import NamedTuple

from cr.nimble import arr_l2norm, arr_vdot

class PowerIterState(NamedTuple):
    """State for the power iterations algorithm
    """
    # (unnormalized) eigen vector guess
    v: jnp.ndarray
    # eigen value estimate
    old_estimate: float
    # new eigen value estimate
    new_estimate: float
    # number of iterations
    iterations: int

class PowerIterSolution(NamedTuple):
    """Solution of the eigen vector estimate
    """
    v: jnp.ndarray
    """The estimated eigen vector"""
    s : float
    """The estimated eigen value"""
    iterations: int
    """Number of iterations to converge"""



def power_iterations(
    operator,
    b,
    max_iters=100,
    error_tolerance=1e-6):
    """Computes the largest eigen value of a (symmetric) linear operator by power method

    Args:
        operator (cr.sparse.lop.Operator): A symmetric linear operator :math:`A`
        b (jax.numpy.ndarray): A user provided initial guess for the largest eigen vector
        max_iters (int): Maximum number of iterations
        error_tolerance (float): Tolerance for relative change in largest eigen value 

    Returns:
        PowerIterSolution: A named tuple containing the largest eigen value, 
        corresponding eigen vector and the number of iterations for convergence

    The operator may accept multi-dimensional arrays as input. E.g. a 2D 
    convolution operator will accept 2D images as input. In such cases, 
    the eigen vector will also be a multi-dimensional array.

    Note:
        This will not work for matrices with complex eigen values.
    """
    def init():
        return PowerIterState(v=b, old_estimate=-1e20, new_estimate=1e20, iterations=0)

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
        v_norm = arr_l2norm(v)
        v = v / v_norm
        # compute the next vector
        v_new = operator.times(v)
        # estimate the eigen value
        new_estimate = jnp.vdot(v, v_new)
        return PowerIterState(v=v_new, old_estimate=state.new_estimate, 
            new_estimate=new_estimate, iterations=state.iterations+1)

    state = lax.while_loop(cond, body, init())
    # We have converged
    v  = state.v
    # normalize the eigen vector again
    v = v / arr_l2norm(v)
    return PowerIterSolution(v = v, s=state.new_estimate, iterations=state.iterations)

power_iterations_jit = jit(power_iterations, static_argnums=(0, 2, 3))
