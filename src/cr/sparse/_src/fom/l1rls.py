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

def l1rls(A, b, lambda_, x0, options: FomOptions = FomOptions()):
    r"""Solver for l1 regulated least square problem

    Args:
        A (cr.sparse.lop.Operator): A linear operator 
        b (jax.numpy.ndarray): The measurements :math:`b \approx A x`
        lambda_ (float): The regularization parameter for the l1 term
        x0 (jax.numpy.ndarray): Initial guess for solution vector
        options (FomOptions): Options for configuring the algorithm

    Returns: 
        FomState: Solution of the optimization problem


    The l1 regularized least square problem is defined as: 

    .. math::

        \text{minimize} \frac{1}{2} \| A x - b \|_2^2 + \lambda \| x \|_1 

    Sometimes, this is also called LASSO in literature.
    """
    f = opt.smooth_quad_matrix()
    h = opt.prox_l1(lambda_)
    return fom(f, h, A, -b, x0, options)


l1rls_jit = jit(l1rls, static_argnums=(0, 4))