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

"""
First Order Conic Solver for Smooth Conic Dual Problem
"""

from .defs import FomOptions, FomState


import jax.numpy as jnp
from jax import jit, lax

import cr.nimble as cnb
import cr.sparse.opt as opt
import cr.sparse.lop as lop


def focs_scd(prox_f, conj_neg_h, A, b, mu, x0, z0, options: FomOptions = FomOptions()):
    r"""First order conic solver for smooth conic dual problems driver routine

    Args:
        prox_f (cr.sparse.opt.SmoothFunction): A prox-capable objective function 
        conj_neg_h (cr.sparse.opt.ProxCapable): The conjugate negative :math:`h^{-}` function 
        A (cr.sparse.lop.Operator): A linear operator 
        b (jax.numpy.ndarray): The translation vector
        mu (float): The (positive) scaling term for the quadratic term :math:`\frac{\mu}{2} \| x - x_0 \|_2^2` 
        x0 (jax.numpy.ndarray): The center point for the quadratic term
        z0 (jax.numpy.ndarray): The initial dual point 
        options (FomOptions): Options for configuring the algorithm

    Returns: 
        FomState: Solution of the optimization problem

    The function uses first order conic solver algorithms to solve an
    optimization problem of the form:
    """
