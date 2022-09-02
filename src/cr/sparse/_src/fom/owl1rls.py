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

def owl1rls(A, b, lambda_, x0, options: FomOptions = FomOptions()):
    r"""Solver for ordered weighted l1 norm regulated least square problem

    Args:
        A (cr.sparse.lop.Operator): A linear operator 
        b (jax.numpy.ndarray): The measurements :math:`b \approx A x`
        lambda_ (jax.numpy.ndarray): A strictly positive weight vector which is sorted in decreasing order
        x0 (jax.numpy.ndarray): Initial guess for solution vector
        options (FomOptions): Options for configuring the algorithm

    Returns: 
        FomState: Solution of the optimization problem


    The ordered weighted l1 regularized least square problem :cite:`lgorzata2013statistical` is defined as:

    .. math::

        \underset{x \in \RR^n}{\text{minimize}} \frac{1}{2} \| A x - b \|_2^2 + \sum_{i=1}^n \lambda_i | x |_{(i)} 


    The ordered weighted :math:`\ell_1` norm of :math:`x` w.r.t. the weight vector :math:`\lambda` is defined as:

    .. math::

        J_{\lambda} (x) = \sum_{1}^n \lambda_i | x |_{(i)}

    See Also:
        :func:`cr.sparse.opt.prox_owl1` for details about the ordered weighted l1 norm.
    """
    f = opt.smooth_quad_matrix()
    h = opt.prox_owl1(lambda_)
    return fom(f, h, A, -b, x0, options)


owl1rls_jit = jit(owl1rls, static_argnums=(0, 4))