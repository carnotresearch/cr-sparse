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

from jax import jit

import cr.sparse.opt as opt


from .util import matrix_affine_func
from .fom import fom
from .defs import FomOptions

def lasso(A, b, tau, x0, options: FomOptions = FomOptions()):
    r"""Solver for LASSO problem

    Args:
        A (cr.sparse.lop.Operator): A linear operator 
        b (jax.numpy.ndarray): The measurements :math:`b \approx A x`
        tau (float): The radius of the l1-ball constraint
        x0 (jax.numpy.ndarray): Initial guess for solution vector
        options (FomOptions): Options for configuring the algorithm

    Returns: 
        FomState: Solution of the optimization problem


    The LASSO problem is defined as: 
    
    .. math::

        \begin{aligned}
        \underset{x}{\text{minimize}} \frac{1}{2} \| \AAA x - b \|_2^2\\
        \text{subject to } \| x \|_1 \leq \tau
        \end{aligned}

    """
    f = opt.smooth_quad_matrix()
    h = opt.prox_l1_ball(tau)
    return fom(f, h, A, -b, x0, options)


lasso_jit = jit(lasso, static_argnums=(0, 4))