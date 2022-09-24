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

import math
from jax import lax
from typing import NamedTuple, List, Dict
import jax.numpy as jnp
from jax.numpy.linalg import norm
import cr.nimble as cnb
import cr.nimble.dsp as crdsp

class MPState(NamedTuple):
    """State of matching pursuit algorithm
    """
    # The non-zero values
    x: jnp.ndarray
    """Non-zero values"""
    r: jnp.ndarray
    """The residuals"""
    r_norm_sqr: float
    """The residual norm squared"""
    iterations: int

    @property
    def I(self):
        return crdsp.support(self.x)

    @property
    def x_I(self):
        return crdsp.nonzero_values(self.x)

    @property
    def length(self):
        return self.x.size

    def __str__(self):
        """Returns the string representation of the state
        """
        s = []
        r_norm = math.sqrt(float(self.r_norm_sqr))
        x_norm = float(norm(self.x))
        for x in [
            f'iterations={self.iterations}',
            f"m={len(self.r)}, n={self.length}, k={len(self.I)}",
            f'r_norm={r_norm:e}',
            f'x_norm={x_norm:e}',
            ]:
            s.append(x.rstrip())
        return u'\n'.join(s)


def solve(Phi, y, max_iters=100, res_norm_rtol=1e-4):
    r"""Solves the sparse recovery problem :math:`y = \Phi x + e`
    using matching pursuit algorithm

    Args:
        Phi: A linear operator
        y(jax.numpy.ndarray): Measurements
        max_iters(int): Maximum number of iterations
        res_norm_rtol(float): Acceptable residual norm

    Returns:
        MPState: A named tuple containing matching pursuit solution state

    MPState interface is similar to
    :py:class:`cr.sparse.pursuit.RecoverySolution`.

    Examples:
        - :ref:`gallery:cs:mp:1`

    Note:
        If the expected sparsity of the solution is known
        then, one should set max_iters as about 4 times
        that value.
    """
    ## Initialize some constants for the algorithm
    m, n = Phi.shape

    # squared norm of the signal
    y_norm_sqr = y.T @ y
    # limit on r norm square
    max_r_norm_sqr = y_norm_sqr * (res_norm_rtol ** 2)

    zero_vec = jnp.zeros(n)
    def init_func():
        # initialize residual
        r = y
        x = jnp.zeros(n)
        return MPState(x=x, r=r, 
            r_norm_sqr=y_norm_sqr, 
            iterations=0)

    def body_func(state):
        h = Phi.trans(state.r)
        abs_h = jnp.abs(h)
        # find the maximum in the column
        best_match_index = jnp.argmax(abs_h)
        # pick corresponding correlation value
        coeff = h[best_match_index]
        # update the representation
        x  = state.x.at[best_match_index].add(coeff)
        # find the best match atom
        atom = Phi.times(zero_vec.at[best_match_index].set(1))
        # update the residual
        r = state.r - coeff * atom
        # Compute residual norm squared
        r_norm_sqr = r.T @ r
        return MPState(x=x, r=r, 
            r_norm_sqr=y_norm_sqr, 
            iterations=state.iterations+1)

    def cond_func(state):
        # limit on residual norm 
        a = state.r_norm_sqr > max_r_norm_sqr
        # limit on number of iterations
        b = state.iterations < max_iters
        c = jnp.logical_and(a, b)
        return c

    # state = init_func()
    # while cond_func(state):
    #     state = body_func(state)
    state = lax.while_loop(cond_func, body_func, init_func())
    return state


def matrix_solve(Phi, y, max_iters=100, res_norm_rtol=1e-4):
    r"""Solves the sparse recovery problem :math:`y = \Phi x + e`
    using matching pursuit algorithm

    Args:
        Phi(jax.numpy.ndarray): A sensing matrix / dictionary
        y(jax.numpy.ndarray): Measurements
        max_iters(int): Maximum number of iterations
        res_norm_rtol(float): Acceptable residual norm

    Returns:
        MPState: A named tuple containing matching pursuit solution state

    MPState interface is similar to
    :py:class:`cr.sparse.pursuit.RecoverySolution`.

    Examples:
        - :ref:`gallery:cs:mp:1`

    Note:
        If the expected sparsity of the solution is known
        then, one should set max_iters as about 4 times
        that value.
    """
    ## Initialize some constants for the algorithm
    m, n = Phi.shape

    # squared norm of the signal
    y_norm_sqr = y.T @ y
    # limit on r norm square
    max_r_norm_sqr = y_norm_sqr * (res_norm_rtol ** 2)

    zero_vec = jnp.zeros(n)
    def init_func():
        # initialize residual
        r = y
        x = jnp.zeros(n)
        return MPState(x=x, r=r, 
            r_norm_sqr=y_norm_sqr, 
            iterations=0)

    def body_func(state):
        h = Phi.T @ state.r
        abs_h = jnp.abs(h)
        # find the maximum in the column
        best_match_index = jnp.argmax(abs_h)
        # pick corresponding correlation value
        coeff = h[best_match_index]
        # update the representation
        x  = state.x.at[best_match_index].add(coeff)
        # find the best match atom
        atom = Phi[:, best_match_index]
        # update the residual
        r = state.r - coeff * atom
        # Compute residual norm squared
        r_norm_sqr = r.T @ r
        return MPState(x=x, r=r, 
            r_norm_sqr=y_norm_sqr, 
            iterations=state.iterations+1)

    def cond_func(state):
        # limit on residual norm 
        a = state.r_norm_sqr > max_r_norm_sqr
        # limit on number of iterations
        b = state.iterations < max_iters
        c = jnp.logical_and(a, b)
        return c

    # state = init_func()
    # while cond_func(state):
    #     state = body_func(state)
    state = lax.while_loop(cond_func, body_func, init_func())
    return state
