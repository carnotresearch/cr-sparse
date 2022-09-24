# Copyright 2022 CR-Suite Development Team
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
import cr.nimble as crn
import cr.nimble.dsp as crdsp


def step_noiseless(Phi, y, x, p=1.):
    """A step in the FOCUSS algorithm

    FOCUSS (FOcal Underdetermined System Solver)
    uses an iterated reweighted least squares method.
    The algorithm is described in :cite:`elad2010sparse` (section 3.2.1).

    Examples:
        - :ref:`gallery:focuss:1`
    """
    xp = jnp.abs(x) ** p
    xp2 = xp * xp
    A = crn.diag_premultiply(xp2, Phi.T)
    G = Phi @ A 
    x = A @ jnp.linalg.solve(G, y)
    return x

class FocussState(NamedTuple):
    """State of iterative reweighted least squares algorithm
    """
    # The non-zero values
    x: jnp.ndarray
    "Solution vector"
    r: jnp.ndarray
    "The residuals"
    r_norm_sqr: float
    "The residual norm squared"
    iterations: int
    "Number of iterations"

    @property
    def length(self):
        return self.x.size

    @property
    def I(self):
        signal, mask = crdsp.energy_threshold(self.x, 0.99)
        return jnp.where(mask)[0]

    @property
    def x_I(self):
        I = self.I
        return self.x[I]

    def __str__(self):
        """Returns the string representation
        """
        s = []
        r_norm = math.sqrt(float(self.r_norm_sqr))
        x_norm = float(norm(self.x))
        I = self.I
        for x in [
            f'iterations={self.iterations}',
            f"m={len(self.r)}, n={self.length}, k={len(I)}",
            f'r_norm={r_norm:e}',
            f'x_norm={x_norm:e}',
            ]:
            s.append(x.rstrip())
        return u'\n'.join(s)


def matrix_solve_noiseless(Phi, y, p=1., max_iters=20):
    """Solves the sparse recovery problem using FOCUSS.

    Args:
        Phi(jax.numpy.ndarray): A sensing matrix / dictionary
        y(jax.numpy.ndarray): Measurements
        max_iters(int): Maximum number of iterations
        p(float): norm type [ p <= 1.]

    FOCUSS (FOcal Underdetermined System Solver)
    uses an iterated reweighted least squares method.
    The algorithm is described in :cite:`elad2010sparse` (section 3.2.1).

    Examples:
        - :ref:`gallery:focuss:1`
    """
    m, n = Phi.shape
    # squared norm of the signal
    y_norm_sqr = y.T @ y

    def init_func():
        # initial solution
        x = jnp.ones(n)
        xp = x
        # initialize residual
        r = y - Phi @ x
        r_norm_sqr = r.T @ r
        return FocussState(x=x, r=r, 
            r_norm_sqr=r_norm_sqr, 
            iterations=0)

    def body_func(state):
        xp = jnp.abs(state.x) ** p
        xp2 = xp * xp
        A = crn.diag_premultiply(xp2, Phi.T)
        G = Phi @ A 
        x = A @ jnp.linalg.solve(G, y)
        r = y - Phi @ x
        r_norm_sqr = r.T @ r
        return FocussState(x=x, r=r, 
            r_norm_sqr=r_norm_sqr, 
            iterations=state.iterations +1)

    def cond_func(state):
        # limit on number of iterations
        b = state.iterations < max_iters
        return b
    # state = init_func()
    # while cond_func(state):
    #     state = body_func(state)
    state = lax.while_loop(cond_func, body_func, init_func())
    return state
