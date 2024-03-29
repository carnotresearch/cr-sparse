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

"""
First Order Solver for the basis pursuit problem
"""

from jax import jit

import cr.sparse.opt as opt
import cr.sparse.lop as lop

from .scd import scd
from .defs import FomOptions


def bp_scd(A, b, mu, x0, z0, options: FomOptions = FomOptions()):
    r"""Solver for the (smoothed) basis pursuit problem using smoothed conic dual formulation


    Args:
        A (cr.sparse.lop.Operator): A linear operator 
        b (jax.numpy.ndarray): The measurements :math:`b \approx A x`
        mu (float): The (positive) scaling term for the quadratic term :math:`\frac{\mu}{2} \| x - x_0 \|_2^2` 
        x0 (jax.numpy.ndarray): The center point for the quadratic term
        z0 (jax.numpy.ndarray): The initial dual point 
        options (FomOptions): Options for configuring the algorithm

    Returns: 
        FomState: Solution of the optimization problem

    We consider the optimization problem 

    .. math::

        \begin{aligned}
            & \underset{x}{\text{minimize}} 
            & &  \| x \|_1  + \frac{\mu}{2} \| x - x_0 \|_2^2\\
            & \text{subject to}
            & &  A x = b
        \end{aligned}
    """
    prox_f = opt.prox_l1()
    conj_neg_h = opt.prox_zero()
    sol = scd(prox_f, conj_neg_h, A, -b, mu, x0, z0, options)
    return sol
